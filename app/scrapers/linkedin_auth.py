"""LinkedIn authenticated session management (for the Voyager API).

Mirrors ``glints_auth``: the backend reuses one authenticated session,
materialized as a Playwright ``storage_state`` (cookies incl. ``li_at`` +
``JSESSIONID``, plus localStorage). This module is the single source of truth
for that session:

* :func:`ensure_session` — pipeline entry point: returns a valid storage_state
  path, auto-minting one via a headless login when missing and credentials are
  set. No manual seeding.
* :func:`storage_state_path` — the path to inject into ``browser.new_context``
  iff a valid file already exists, else ``None`` (no login attempted).
* :func:`refresh_session` — forces a headless credential login that (re-)mints
  the storage_state; used reactively when an in-flight session expires.

With no credentials everything degrades to ``None`` and LinkedIn scraping stays
on the anonymous guest endpoint. WARNING: authenticated Voyager use violates
LinkedIn's ToS and may get the account checkpoint-challenged or banned.
"""

import asyncio
import json
import re
from pathlib import Path

from loguru import logger

from app.config import settings
from app.scrapers.base import apply_stealth, humanizer_delay, random_user_agent

_LOGIN_URL = "https://www.linkedin.com/login"
# LinkedIn's newer React login UI uses obfuscated ids/classes and no <form> or
# submit button, so we anchor on the stable semantic attributes (autocomplete=…)
# first and keep the classic ids as fallbacks for the legacy layout.
_EMAIL_INPUT = (
    "input[autocomplete='username'], input#username, input[name='session_key'], input[type='email']"
)
_PASSWORD_INPUT = (
    "input[autocomplete='current-password'], input#password, "
    "input[name='session_password'], input[type='password']"
)
# The new UI's submit is a <button type="button"> (no <form>/submit) whose text
# is "Login"/"Masuk"/"Sign in". Anchored so it never matches the social-login
# buttons ("Masuk dengan Microsoft", "Continue with Google", …).
_SUBMIT_NAME_RE = re.compile(r"^\s*(log\s?in|masuk|sign\s?in)\s*$", re.I)
# After a successful login LinkedIn lands on /feed. A checkpoint/challenge/2FA
# redirects to one of these path fragments instead — we detect and bail.
_CHECKPOINT_FRAGMENTS = ("/checkpoint/", "/uas/", "/challenge/")

# Serialize re-logins so parallel scrapes don't all hammer the login form.
_refresh_lock = asyncio.Lock()


def storage_state_path() -> str | None:
    """Return the configured storage_state path iff the file exists and parses
    as JSON; otherwise ``None``. Callers pass the result straight to
    ``browser.new_context(storage_state=...)``."""
    path = settings.linkedin_storage_state_path
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        json.loads(p.read_text())
    except (ValueError, OSError):
        logger.warning(f"linkedin_auth: storage_state at {path} is unreadable/invalid")
        return None
    return path


def has_credentials() -> bool:
    return bool(settings.linkedin_email and settings.linkedin_password)


async def refresh_session() -> str | None:
    """Headless login with the configured credentials; persist a fresh
    storage_state. Returns the saved path, or ``None`` if credentials are unset
    or login failed. Concurrent callers are serialized."""
    if not has_credentials():
        logger.info(
            "linkedin_auth: no LINKEDIN_EMAIL/LINKEDIN_PASSWORD set; cannot "
            "refresh session (guest endpoint only)."
        )
        return None
    async with _refresh_lock:
        return await _login_and_save()


async def ensure_session() -> str | None:
    """Return a valid storage_state path, seeding one on first use:

    1. A valid session file exists -> return it (no login).
    2. No file but credentials set -> headless login, write + validate, return.
    3. No file and no credentials -> ``None`` (guest endpoint only).
    """
    existing = storage_state_path()
    if existing is not None:
        return existing
    if not has_credentials():
        logger.info(
            "linkedin_auth: no saved session and no LINKEDIN_EMAIL/PASSWORD; "
            "scraping LinkedIn via the anonymous guest endpoint."
        )
        return None
    async with _refresh_lock:
        existing = storage_state_path()
        if existing is not None:
            return existing
        logger.info(
            "linkedin_auth: no saved session; performing initial headless login "
            "from .env credentials (unattended seed)…"
        )
        if await _login_and_save() is None:
            return None
        validated = storage_state_path()
        if validated is None:
            logger.warning(
                "linkedin_auth: initial login reported success but the state file "
                "did not validate; continuing on the guest endpoint."
            )
        return validated


def _is_checkpoint_url(url: str) -> bool:
    """True when ``url`` is a LinkedIn checkpoint/challenge/2FA page — the signal
    that headless login was intercepted and we must bail (→ guest fallback)."""
    u = (url or "").lower()
    return any(frag in u for frag in _CHECKPOINT_FRAGMENTS)


def _diagnose_login_page(url: str, title: str, body: str) -> str:
    """Best-effort classification of the page LinkedIn served when the login form
    didn't appear / login didn't complete, so the log says WHY rather than just
    "form never appeared". Pure (text-only) so it's unit-testable."""
    hay = f"{url}\n{title}\n{body}".lower()
    checks = [
        (
            (
                "captcha",
                "px-captcha",
                "unusual activity",
                "unusual traffic",
                "press & hold",
                "verify you're",
                "are you a human",
            ),
            "anti-bot/CAPTCHA wall",
        ),
        (
            (
                "two-step",
                "verification code",
                "enter the code",
                "security check",
                "/checkpoint/",
                "we sent a code",
            ),
            "checkpoint/2FA page",
        ),
        (
            (
                "we value your privacy",
                "accept cookies",
                "cookie policy",
                "manage preferences",
                "consent",
            ),
            "cookie-consent interstitial",
        ),
        (
            ("join now", "new to linkedin", "make the most of your professional life"),
            "signup/marketing splash (not the login form)",
        ),
    ]
    for kws, label in checks:
        if any(k in hay for k in kws):
            return label
    return "unrecognized page (no login-form selector matched)"


async def _report_login_failure(page, reason: str) -> None:
    """Capture and log WHAT LinkedIn actually served when login couldn't proceed:
    a classified diagnosis of the served page. Best-effort and never raises — it
    only adds visibility to an already-failed login."""
    try:
        url = page.url
        title = await page.title()
        body = await page.content()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"linkedin_auth: {reason}; (could not read served page: {e})")
        return
    diagnosis = _diagnose_login_page(url, title, body)
    logger.warning(f"linkedin_auth: {reason} → {diagnosis} (url={url!r}, title={title!r})")


async def _first_visible(page, selector: str):
    """Return the first VISIBLE element matching ``selector``, or None. LinkedIn's
    new login renders duplicate (e.g. desktop+mobile) input variants where only
    one set is visible, so a plain selector can latch onto the hidden one."""
    loc = page.locator(selector)
    for i in range(await loc.count()):
        el = loc.nth(i)
        try:
            if await el.is_visible():
                return el
        except Exception:  # noqa: BLE001
            continue
    return None


async def _fill_humanized(page, loc, text: str) -> None:
    """Type ``text`` into the given locator with humanized per-keystroke delay."""
    await loc.click()
    await page.wait_for_timeout(int(humanizer_delay("click") * 1000))
    await loc.press_sequentially(text, delay=int(humanizer_delay("keystroke") * 1000))


async def _click_submit(page, password_el) -> None:
    """Submit the login. The new UI has no <form>/submit input — the sign-in
    control is a ``<button type="button">`` whose text is Login/Masuk (duplicated,
    one variant hidden), so we match by accessible name and click the visible one.
    Falls back to pressing Enter on the password field if no such button is found."""
    try:
        btn = page.get_by_role("button", name=_SUBMIT_NAME_RE)
        for i in range(await btn.count()):
            el = btn.nth(i)
            if await el.is_visible():
                await el.click()
                return
    except Exception:  # noqa: BLE001 — fall back to Enter below
        pass
    await password_el.press("Enter")


async def _login_and_save() -> str | None:
    from playwright.async_api import async_playwright

    path = settings.linkedin_storage_state_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = None
        context = None
        try:
            browser = await pw.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                user_agent=random_user_agent(),
            )
            page = await context.new_page()
            await apply_stealth(page)

            await page.goto(_LOGIN_URL, timeout=30000, wait_until="domcontentloaded")
            try:
                await page.wait_for_selector(_EMAIL_INPUT, timeout=15000)
            except Exception:
                await _report_login_failure(
                    page,
                    "login form never appeared (layout changed or anti-bot "
                    "block); aborting → guest fallback",
                )
                return None

            # The new UI duplicates inputs (one variant hidden); target the visible
            # ones, and submit with Enter since there is no <form>/submit button.
            email_el = await _first_visible(page, _EMAIL_INPUT)
            password_el = await _first_visible(page, _PASSWORD_INPUT)
            if email_el is None or password_el is None:
                await _report_login_failure(
                    page,
                    "login inputs present but not visible (new login UI "
                    "variant?); aborting → guest fallback",
                )
                return None

            await _fill_humanized(page, email_el, settings.linkedin_email)
            await page.wait_for_timeout(int(humanizer_delay("click") * 1000))
            await _fill_humanized(page, password_el, settings.linkedin_password)
            await page.wait_for_timeout(int(humanizer_delay("click") * 1000))
            await _click_submit(page, password_el)

            # Success = navigation away from /login to an authenticated view, and
            # NOT onto a checkpoint/challenge page.
            try:
                await page.wait_for_url(lambda u: "/login" not in u, timeout=20000)
            except Exception:
                await _report_login_failure(
                    page,
                    "login did not navigate within 20s (credentials rejected "
                    "or anti-bot); aborting → guest fallback",
                )
                return None

            if _is_checkpoint_url(page.url):
                logger.warning(
                    f"linkedin_auth: login hit a checkpoint/2FA page ({page.url}); "
                    "headless login cannot proceed → guest fallback. Seed a session "
                    "from a desktop browser if you need authenticated Voyager."
                )
                return None

            await context.storage_state(path=path)
            logger.info(f"linkedin_auth: refreshed LinkedIn session -> {path}")
            return path
        except Exception as e:  # noqa: BLE001
            logger.warning(f"linkedin_auth: headless login failed: {e}")
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
