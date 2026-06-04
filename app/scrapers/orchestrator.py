import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from app.config import settings
from app.schemas import JobListingDTO, SearchParams
from app.scrapers import glints_auth, linkedin_auth
from app.scrapers.base import BaseScraper, browser_session
from app.scrapers.glints import GlintsScraper
from app.scrapers.linkedin import LinkedInScraper

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]

_REGISTRY: dict[str, type[BaseScraper]] = {
    "glints": GlintsScraper,
    "linkedin": LinkedInScraper,
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def dedupe_by_company_title(jobs: list[JobListingDTO]) -> list[JobListingDTO]:
    seen: set[tuple[str, str]] = set()
    out: list[JobListingDTO] = []
    for j in jobs:
        key = (_norm(j.company), _norm(j.title))
        if key in seen:
            continue
        seen.add(key)
        out.append(j)
    return out


async def run_portals(
    portals: list[str],
    params: SearchParams,
    on_event: EventEmitter,
) -> list[JobListingDTO]:
    async def _run_one(name: str) -> list[JobListingDTO]:
        cls = _REGISTRY.get(name)
        if cls is None:
            await on_event(
                {
                    "type": "error",
                    "severity": "warning",
                    "portal": name,
                    "message": f"unknown portal {name}",
                }
            )
            return []
        scraper = cls()
        await on_event({"type": "portal_start", "portal": name})
        # Seed the context with a portal-specific authenticated session where one
        # is configured (Glints pages 2+ are login-gated; LinkedIn uses it for the
        # Voyager API). ensure_session auto-mints via headless login on first use
        # when credentials are set, else returns None (anonymous/guest path).
        if name == "glints":
            storage_state = await glints_auth.ensure_session()
        elif name == "linkedin":
            storage_state = await linkedin_auth.ensure_session()
        else:
            storage_state = None
        try:
            async with browser_session(name, storage_state=storage_state) as page:
                # Scrapers self-time-out cooperatively (see GlintsScraper's
                # deadline). wait_for here is just a hard backstop for genuine
                # hangs (e.g. stuck Playwright IPC); the grace ensures the
                # scraper's own deadline fires first so we never throw away
                # already-scraped jobs to a TimeoutError.
                jobs = await asyncio.wait_for(
                    scraper.scrape(page, params, on_event),
                    timeout=settings.scraper_timeout_seconds + 15,
                )
        except (TimeoutError, Exception) as e:
            await on_event(
                {
                    "type": "error",
                    "severity": "warning",
                    "portal": name,
                    "message": f"{type(e).__name__}: {e}",
                }
            )
            return []
        await on_event({"type": "portal_complete", "portal": name})
        return jobs

    chunks = await asyncio.gather(*(_run_one(p) for p in portals))
    flat = [j for chunk in chunks for j in chunk]
    return dedupe_by_company_title(flat)
