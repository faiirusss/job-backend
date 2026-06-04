"""Seed a LinkedIn session for the scraper from manually-copied cookies.

LinkedIn's headless login is walled off by an obfuscated React UI + anti-bot, so
the reliable way to get an authenticated session (needed for the Voyager API) is
to copy two cookies from a browser you're already logged into:

  - ``li_at``      — the auth token
  - ``JSESSIONID`` — also the source of LinkedIn's ``csrf-token``

In your browser: DevTools → Application → Cookies → ``https://www.linkedin.com``,
copy those two values, then::

    python -m app.scrapers.linkedin_login --li-at "AQED..." --jsessionid "ajax:123..."

Writes a Playwright ``storage_state`` to ``LINKEDIN_STORAGE_STATE_PATH`` (default
``./data/linkedin_state.json``), which :mod:`app.scrapers.linkedin_auth` then
reuses. Gitignored — never commit it.
"""

import argparse
import json
from pathlib import Path

from app.config import settings

# Cookie expiry (Y2038). li_at itself lasts ~1 year; this just avoids a session
# cookie that Playwright would drop on context close.
_FAR_FUTURE = 2147483647


def _normalize_jsessionid(value: str) -> str:
    """LinkedIn stores JSESSIONID wrapped in double quotes; ensure that exact
    on-the-wire form regardless of whether the user pasted it quoted or bare.
    ``csrf_token_from_state`` strips the quotes again for the header."""
    return '"' + value.strip().strip('"') + '"'


def build_storage_state(li_at: str, jsessionid: str) -> dict:
    """Build a Playwright storage_state dict carrying the two LinkedIn auth
    cookies on the domains the Voyager endpoints need. Raises ValueError if
    either value is empty."""
    li_at = li_at.strip()
    if not li_at or not jsessionid.strip():
        raise ValueError("both li_at and jsessionid are required")
    return {
        "cookies": [
            {
                "name": "li_at",
                "value": li_at,
                "domain": ".linkedin.com",
                "path": "/",
                "expires": _FAR_FUTURE,
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
            },
            {
                "name": "JSESSIONID",
                "value": _normalize_jsessionid(jsessionid),
                "domain": ".www.linkedin.com",
                "path": "/",
                "expires": _FAR_FUTURE,
                "httpOnly": False,
                "secure": True,
                "sameSite": "None",
            },
        ],
        "origins": [],
    }


def write_storage_state(li_at: str, jsessionid: str, out_path: str) -> str:
    state = build_storage_state(li_at, jsessionid)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(state, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed a LinkedIn scraper session from li_at + JSESSIONID cookies."
    )
    parser.add_argument("--li-at", required=True, help="The li_at cookie value.")
    parser.add_argument(
        "--jsessionid", required=True, help="The JSESSIONID cookie value (quoted or bare)."
    )
    parser.add_argument(
        "--out",
        default=settings.linkedin_storage_state_path,
        help="Output storage_state path (default: LINKEDIN_STORAGE_STATE_PATH).",
    )
    args = parser.parse_args()
    out = write_storage_state(args.li_at, args.jsessionid, args.out)
    print(f"Wrote LinkedIn session -> {out}")


if __name__ == "__main__":
    main()
