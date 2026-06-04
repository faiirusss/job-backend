"""Seed a Glints service-account session for the scraper.

Glints gates search pages 2+ behind login. The scraper reuses a persisted
Playwright ``storage_state`` (see :mod:`app.scrapers.glints_auth`); this CLI
creates that file. Two modes:

* default — open a **visible** browser, you log in by hand (works with Google
  login / OTP / captcha), then the authenticated session is saved.
* ``--from-cookies FILE`` — build the storage_state from a cookies JSON you
  exported from your normal browser (use this when a headed browser can't run,
  e.g. WSL2 without WSLg).

Usage::

    python -m app.scrapers.glints_login
    python -m app.scrapers.glints_login --from-cookies glints_cookies.json

The session is written to ``GLINTS_STORAGE_STATE_PATH`` (default
``./data/glints_state.json``). It is gitignored — never commit it.
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import settings

_LOGIN_URL = "https://glints.com/id/login"


async def _interactive_login(out_path: str) -> None:
    from playwright.async_api import async_playwright

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()
        await page.goto(_LOGIN_URL, wait_until="domcontentloaded")
        print(
            "\nA browser window is open. Log into Glints (any method), wait until you\n"
            "see your logged-in dashboard, then press Enter here to save the session.\n"
        )
        # Block on stdin without freezing the asyncio loop / browser.
        await asyncio.get_event_loop().run_in_executor(None, input)
        await context.storage_state(path=out_path)
        await context.close()
        await browser.close()
    print(f"Saved Glints session -> {out_path}")


def _from_cookies(cookies_file: str, out_path: str) -> None:
    cookies = json.loads(Path(cookies_file).read_text())
    if isinstance(cookies, dict) and "cookies" in cookies:
        cookies = cookies["cookies"]  # accept a full storage_state too
    if not isinstance(cookies, list):
        raise SystemExit("--from-cookies file must be a JSON array of cookie objects")
    state = {"cookies": cookies, "origins": []}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(state, indent=2))
    print(f"Wrote storage_state from {len(cookies)} cookies -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a Glints scraper session.")
    parser.add_argument(
        "--from-cookies",
        metavar="FILE",
        help="Build storage_state from an exported cookies JSON instead of a headed login.",
    )
    parser.add_argument(
        "--out",
        default=settings.glints_storage_state_path,
        help="Output storage_state path (default: GLINTS_STORAGE_STATE_PATH).",
    )
    args = parser.parse_args()

    if args.from_cookies:
        _from_cookies(args.from_cookies, args.out)
    else:
        asyncio.run(_interactive_login(args.out))


if __name__ == "__main__":
    main()
