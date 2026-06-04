"""LinkedIn Voyager API helpers + a fire-and-dump capture step.

Phase 1 deliberately does NOT normalize Voyager responses — it captures them.
The request shape below was confirmed from a real authenticated browser request:
job search is the Voyager **GraphQL** endpoint with a versioned ``queryId`` and a
RestLi tuple ``variables=(...)``, and Voyager requires the ``normalized+json``
Accept type plus ``x-li-track`` / ``x-li-page-instance`` (plain application/json
gets HTTP 400).

Version note: ``_JOB_SEARCH_QUERY_ID`` and ``_CLIENT_VERSION`` are pinned to a
LinkedIn web build and rotate with releases — if search starts returning 400,
re-capture them from a browser (DevTools → Network → the graphql jobCards call).
"""

import json
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from loguru import logger

_VOYAGER_BASE = "https://www.linkedin.com/voyager/api"
_GRAPHQL_URL = f"{_VOYAGER_BASE}/graphql"
# Pinned to a LinkedIn web build (captured 2026-06-01); refresh from a browser if
# search 400s. The detail endpoint is still REST and unconfirmed (Phase 2).
_JOB_SEARCH_QUERY_ID = "voyagerJobsDashJobCards.909b0d446794dad30bb8a39a7f8997a4"
_CLIENT_VERSION = "1.13.44480"
_DETAIL_PATH = "jobs/jobPostings"


def csrf_token_from_state(storage_state_path: str) -> str | None:
    """Read the JSESSIONID cookie value out of a Playwright storage_state file
    and return it with surrounding quotes stripped — LinkedIn's csrf-token."""
    try:
        data = json.loads(Path(storage_state_path).read_text())
    except (ValueError, OSError):
        return None
    for c in data.get("cookies", []):
        if c.get("name") == "JSESSIONID":
            return str(c.get("value", "")).strip().strip('"') or None
    return None


def voyager_headers(storage_state_path: str) -> dict[str, str]:
    """Headers required by Voyager. Empty dict when no csrf-token is available
    (caller then skips the Voyager attempt). Cookies travel via the authenticated
    browser context, not these headers. The Accept type and x-li-* headers are
    required — plain application/json yields HTTP 400."""
    token = csrf_token_from_state(storage_state_path)
    if not token:
        return {}
    track = json.dumps(
        {
            "clientVersion": _CLIENT_VERSION,
            "mpVersion": _CLIENT_VERSION,
            "osName": "web",
            "timezoneOffset": 7,
            "timezone": "Asia/Jakarta",
            "deviceFormFactor": "DESKTOP",
            "mpName": "voyager-web",
            "displayDensity": 1,
            "displayWidth": 1920,
            "displayHeight": 1080,
        },
        separators=(",", ":"),
    )
    return {
        "csrf-token": token,
        "accept": "application/vnd.linkedin.normalized+json+2.1",
        "x-restli-protocol-version": "2.0.0",
        "x-li-lang": "en_US",
        "x-li-track": track,
        "x-li-page-instance": f"urn:li:page:d_flagship3_job_home;{uuid.uuid4()}",
    }


def voyager_detail_url(job_id: str) -> str:
    return f"{_VOYAGER_BASE}/{_DETAIL_PATH}/{quote(str(job_id), safe='')}"


def voyager_search_url(keywords: str, location: str = "", start: int = 0, count: int = 25) -> str:
    """Build the Voyager GraphQL job-search URL. Location is folded into the
    keywords term (as LinkedIn's own search button does); the RestLi tuple's
    structure chars stay literal while the keyword value is percent-encoded."""
    terms = " ".join(t for t in ((keywords or "engineer"), location) if t).strip()
    kw = quote(terms, safe="")
    variables = (
        f"(count:{count},query:(origin:JOBS_HOME_SEARCH_BUTTON,keywords:{kw}),start:{start})"
    )
    return (
        f"{_GRAPHQL_URL}?includeWebMetadata=true"
        f"&variables={variables}&queryId={_JOB_SEARCH_QUERY_ID}"
    )


async def capture_voyager(
    page: Any,
    *,
    storage_state_path: str,
    keywords: str,
    location: str,
    job_ids: list[str],
    dump: Any,
    max_details: int = 5,
) -> None:
    """Best-effort: hit the Voyager search + a few detail endpoints and dump every
    raw response (status + body) so we can pin the JSON shape offline. Never
    raises — capture failure must not affect the live (guest) result. No-ops when
    no csrf-token is available."""
    headers = voyager_headers(storage_state_path)
    if not headers:
        logger.info("linkedin_voyager: no csrf-token; skipping Voyager capture")
        return

    async def _fetch(url: str) -> str:
        try:
            resp = await page.request.get(url, headers=headers)
            body = await resp.text()
            return json.dumps({"status": resp.status, "body": body})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"linkedin_voyager: capture fetch failed for {url}: {e}")
            return json.dumps({"status": 0, "error": str(e)})

    try:
        search_url = voyager_search_url(keywords, location, start=0)
        dump.add_listing(1, search_url, await _fetch(search_url))
        for jid in job_ids[:max_details]:
            url = voyager_detail_url(jid)
            dump.add_detail(jid, await _fetch(url))
        logger.info(
            f"linkedin_voyager: captured Voyager search + "
            f"{min(len(job_ids), max_details)} detail payload(s)"
        )
    except Exception as e:  # noqa: BLE001 — capture is best-effort; never break a run
        logger.warning(f"linkedin_voyager: capture_voyager failed: {e}")
