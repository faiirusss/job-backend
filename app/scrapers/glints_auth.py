"""Glints service-account session management.

Glints gates search pages 2+ behind login (HTTP 403 ``NO_PERMISSION``). To gather
more than the anonymous page-1 cap of 30 jobs, the backend reuses a single
authenticated "service account" session, materialized as a Playwright
``storage_state`` (cookies + localStorage, where Glints keeps its JWT). This
module is the single source of truth for that session:

* :func:`ensure_session` — the pipeline entry point: returns a valid
  storage_state path, **auto-minting one on first use** via a headless login
  when the file is missing and credentials are configured. No manual seeding.
* :func:`storage_state_path` — the path to inject into ``browser.new_context``
  iff a valid file already exists, else ``None`` (no login attempted).
* :func:`refresh_session` — forces a headless credential login that (re-)mints
  the storage_state; used reactively when an in-flight session expires.

With no credentials, everything degrades to ``None`` and Glints scraping stays
anonymous (page-1 only) — zero config required. With credentials set, the very
first search seeds ``glints_state.json`` automatically and unattended.
"""

import asyncio
import json
from pathlib import Path

from loguru import logger

from app.config import settings
from app.scrapers.base import apply_stealth, humanizer_delay, random_user_agent

_LOGIN_URL = "https://glints.com/id/login"
_EXPLORE_URL = "https://glints.com/id/opportunities/jobs/explore"

# Live login DOM (audited 2026-05-29). The form is gated behind a Cloudflare
# "Just a moment..." managed challenge and a "Masuk dengan Email" reveal:
#   1. Cloudflare clears on its own (~1-2s) once stealth presents a clean
#      fingerprint — no interactive solve when the browser looks legitimate.
#   2. The login card shows social login + a "Masuk dengan Email" link; clicking
#      it reveals the email/password inputs (ids #login-form-email / -password).
#   3. A <button type="submit"> ("MASUK") submits the form.
_EMAIL_LINK_TEXT = "Masuk dengan Email"
_EMAIL_INPUT = "input#login-form-email, input[type='email'], input[name='email']"
_PASSWORD_INPUT = "input#login-form-password, input[type='password'], input[name='password']"
# The header also has a no-`type` "MASUK" <button>; type='submit' targets the form.
_SUBMIT_BUTTON = "button[type='submit']"

# Serialize re-logins so parallel scrapes don't all hammer the login form at once.
_refresh_lock = asyncio.Lock()


def storage_state_path() -> str | None:
    """Return the configured storage_state path iff the file exists and parses
    as JSON; otherwise ``None``. Callers pass the result straight to
    ``browser.new_context(storage_state=...)``.
    """
    path = settings.glints_storage_state_path
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        json.loads(p.read_text())
    except (ValueError, OSError):
        logger.warning(f"glints_auth: storage_state at {path} is unreadable/invalid")
        return None
    return path


def has_credentials() -> bool:
    return bool(settings.glints_email and settings.glints_password)


async def refresh_session() -> str | None:
    """Log into Glints headlessly with the configured credentials and persist a
    fresh storage_state. Returns the saved path, or ``None`` if credentials are
    unset or login failed. Concurrent callers are serialized; a caller that loses
    the race returns the session the winner just minted.
    """
    if not has_credentials():
        logger.info(
            "glints_auth: no GLINTS_EMAIL/GLINTS_PASSWORD set; cannot refresh "
            "session (anonymous page-1 only). Seed one with `python -m "
            "app.scrapers.glints_login`."
        )
        return None

    # The lock serializes re-logins; only one credential login runs at a time.
    # refresh_session is called only on expiry detection (rare), so we don't try
    # to coalesce queued callers beyond that — each gets a fresh, valid session.
    async with _refresh_lock:
        return await _login_and_save()


async def ensure_session() -> str | None:
    """Return a valid storage_state path for the scraper, seeding one on first
    use. Resolution order:

    1. A valid session file already exists -> return it (no login).
    2. No file but credentials are set -> perform an unattended headless login,
       write + validate ``glints_state.json``, and return its path.
    3. No file and no credentials -> return ``None`` (anonymous page-1 only).

    This is what makes Glints scraping "automated out of the box": the orchestrator
    awaits this at the start of every Glints run, so the first search after
    configuring GLINTS_EMAIL/GLINTS_PASSWORD mints the session with no manual step.
    """
    existing = storage_state_path()
    if existing is not None:
        return existing
    if not has_credentials():
        logger.info(
            "glints_auth: no saved session and no GLINTS_EMAIL/GLINTS_PASSWORD; "
            "scraping Glints anonymously (page-1 only)."
        )
        return None

    # Double-checked locking: a concurrent search may seed the file while we wait.
    async with _refresh_lock:
        existing = storage_state_path()
        if existing is not None:
            return existing
        logger.info(
            "glints_auth: no saved session found; performing initial headless "
            "login from .env credentials (unattended seed)…"
        )
        if await _login_and_save() is None:
            return None
        # Validate what we just wrote actually exists and parses.
        validated = storage_state_path()
        if validated is None:
            logger.warning(
                "glints_auth: initial login reported success but the state file "
                "did not validate; continuing anonymously."
            )
        return validated


async def _await_cloudflare_clear(page, timeout_s: int = 20) -> bool:
    """Poll until the Cloudflare "Just a moment..." managed challenge clears.

    "Cleared" = the document title is no longer the interstitial *or* the login
    form has already rendered. With stealth applied this typically takes ~1-2s;
    if it never clears we treat it as a persistent anti-bot block (the caller
    surfaces a clear warning).
    """
    for _ in range(timeout_s):
        title = (await page.title()).lower()
        if title and "just a moment" not in title:
            return True
        if await page.locator(_EMAIL_INPUT).count():
            return True
        await page.wait_for_timeout(1000)
    return False


async def _fill_humanized(page, selector: str, text: str) -> None:
    """Type ``text`` into ``selector`` with a randomized inter-keystroke delay
    (``press_sequentially``) rather than an instant ``fill``, so the input looks
    human to generic bot heuristics."""
    loc = page.locator(selector).first
    await loc.click()
    await page.wait_for_timeout(int(humanizer_delay("click") * 1000))
    # delay is per-keystroke milliseconds; humanizer_delay("keystroke") is ~80-150ms.
    await loc.press_sequentially(text, delay=int(humanizer_delay("keystroke") * 1000))


async def _diagnose_login_failure(page) -> str:
    """Best-effort classification of why we're still on /login, for the log."""
    try:
        info = await page.evaluate(
            "() => ({ url: location.href, body: document.body.innerText.toLowerCase() })"
        )
    except Exception:
        return "still on /login (could not read page)"
    body = info.get("body", "")
    url = info.get("url", "")
    checks = [
        (
            ("captcha", "just a moment", "unusual traffic", "press & hold"),
            "a CAPTCHA / anti-bot challenge re-appeared after submit",
        ),
        (
            ("verifikasi", "verification", "kode otp", "one-time", "kode verifikasi"),
            "an OTP / 2FA verification step is required",
        ),
        (
            ("salah", "incorrect", "invalid", "tidak valid", "wrong"),
            "credentials were rejected (wrong email/password)",
        ),
    ]
    for keywords, msg in checks:
        if any(k in body for k in keywords):
            return f"{msg} — still on {url}"
    return f"no recognized error; submit did not navigate within 20s — still on {url}"


async def _login_and_save() -> str | None:
    from playwright.async_api import async_playwright

    path = settings.glints_storage_state_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = None
        context = None
        try:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                user_agent=random_user_agent(),
            )
            page = await context.new_page()
            # Stealth must be applied BEFORE navigation so its init scripts run on
            # the first load — that clean fingerprint is what clears Cloudflare.
            await apply_stealth(page)

            await page.goto(_LOGIN_URL, timeout=30000, wait_until="domcontentloaded")

            # Step 1: wait out the Cloudflare managed challenge.
            if not await _await_cloudflare_clear(page):
                logger.warning(
                    "glints_auth: login blocked by a persistent Cloudflare / anti-bot "
                    "challenge (page never left 'Just a moment...'). Headless login "
                    "cannot proceed — seed a session from a desktop browser with "
                    "`python -m app.scrapers.glints_login`."
                )
                return None

            # Step 2: reveal the email/password form (skip iff already present).
            if not await page.locator(_EMAIL_INPUT).count():
                link = page.get_by_text(_EMAIL_LINK_TEXT, exact=False)
                if await link.count():
                    await link.first.click()
                    await page.wait_for_timeout(int(humanizer_delay("click") * 1000))
            try:
                await page.wait_for_selector(_EMAIL_INPUT, state="visible", timeout=10000)
            except Exception:
                logger.warning(
                    "glints_auth: email login form never appeared after the "
                    f"'{_EMAIL_LINK_TEXT}' step (Glints layout changed?). Aborting login."
                )
                return None

            # Step 3: humanized credential entry + submit.
            await _fill_humanized(page, _EMAIL_INPUT, settings.glints_email)
            await page.wait_for_timeout(int(humanizer_delay("click") * 1000))
            await _fill_humanized(page, _PASSWORD_INPUT, settings.glints_password)
            await page.wait_for_timeout(int(humanizer_delay("click") * 1000))
            await page.locator(_SUBMIT_BUTTON).first.click()

            # Success = navigation away from the login page to an authenticated view.
            try:
                await page.wait_for_url(lambda u: "/login" not in u, timeout=20000)
            except Exception:
                reason = await _diagnose_login_failure(page)
                logger.warning(f"glints_auth: login did not complete — {reason}")
                return None

            await context.storage_state(path=path)
            logger.info(f"glints_auth: refreshed Glints session -> {path}")
            return path
        except Exception as e:
            logger.warning(f"glints_auth: headless login failed: {e}")
            return None
        finally:
            try:
                if context is not None:
                    await asyncio.wait_for(context.close(), timeout=10)
            except Exception:
                pass
            try:
                if browser is not None:
                    await asyncio.wait_for(browser.close(), timeout=10)
            except Exception:
                pass
