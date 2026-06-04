import asyncio
import random
from typing import Any
from urllib.parse import urlencode

from loguru import logger

from app.config import settings
from app.schemas import JobListingDTO, SearchParams
from app.scrapers import (
    linkedin_auth,
    linkedin_normalize as ln,
    linkedin_voyager as lv,
    linkedin_voyager_normalize as vn,
)
from app.scrapers.base import BaseScraper, EventEmitter, humanizer_delay

_MAX_PAGES = 5
_MAX_JOBS = 100
# Fraction of the per-portal budget to finish within; the orchestrator's
# wait_for is the hard backstop. Mirrors GlintsScraper.
_ENRICH_BUDGET_FRACTION = 0.85
# LinkedIn's guest detail endpoint rate-limits bursts (429 after ~6-9 requests),
# so we keep the fan-out low and lean on backoff-retry (below) to recover the
# throttled ones within the otherwise-idle budget.
_ENRICH_CONCURRENCY = 3
# 429 retry policy for the detail endpoint. The budget is mostly idle during
# enrichment, so we can afford a few backoff waits; each retry is still bounded
# by the cooperative deadline.
_DETAIL_MAX_RETRIES = 4
_DETAIL_BACKOFF_BASE = 0.5  # seconds; exponential per attempt, plus jitter

_LISTING_BASE = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
_DETAIL_BASE = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting"


def _retry_after_seconds(headers: Any) -> float | None:
    """Parse a Retry-After header (delta-seconds form) if present and numeric."""
    try:
        val = headers.get("retry-after")
    except AttributeError:
        return None
    if not val:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class LinkedInScraper(BaseScraper):
    portal = "linkedin"

    def build_search_url(self, params: SearchParams) -> str:
        keywords = " ".join(params.role_keywords) or "engineer"
        location = " ".join(params.location) or "Indonesia"
        qs = urlencode({"keywords": keywords, "location": location, "start": 0})
        return f"{_LISTING_BASE}?{qs}"

    async def _fetch(self, page: Any, url: str) -> tuple[str | None, int]:
        """GET via the browser context's request stack (shares stealth/UA/cookies)."""
        try:
            resp = await page.request.get(url)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"linkedin: fetch {url} failed: {e}")
            return None, 0
        if resp.status != 200:
            return None, resp.status
        return await resp.text(), 200

    async def scrape(
        self, page: Any, params: SearchParams, on_event: EventEmitter
    ) -> list[JobListingDTO]:
        return await self._scrape_impl(page, params, on_event)

    async def _scrape_impl(
        self,
        page: Any,
        params: SearchParams,
        on_event: EventEmitter,
    ) -> list[JobListingDTO]:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + settings.scraper_timeout_seconds * _ENRICH_BUDGET_FRACTION

        # Authenticated path first: Voyager gives the rich JD (description,
        # employment type, experience level, …). Only fall back to the guest
        # scrape when there is no session or Voyager yields nothing — this avoids
        # running (and then discarding) the slower guest listing scrape on every
        # authenticated search.
        state = linkedin_auth.storage_state_path()
        if state is not None:
            voyager_jobs = await self._scrape_voyager(page, params, state, deadline, on_event)
            if voyager_jobs:
                return voyager_jobs
            logger.info("linkedin: Voyager path yielded nothing; using guest scrape")

        return await self._scrape_guest(page, params, on_event, deadline)

    async def _scrape_guest(
        self,
        page: Any,
        params: SearchParams,
        on_event: EventEmitter,
        deadline: float,
    ) -> list[JobListingDTO]:
        """Anonymous guest-endpoint scrape (listing pages + detail enrichment).
        The fallback when no authenticated Voyager session is available."""
        loop = asyncio.get_event_loop()
        base = self.build_search_url(params)

        all_jobs: list[JobListingDTO] = []
        seen: set[str] = set()
        # The guest endpoint returns a variable, smaller-than-nominal number of
        # cards per call, so we advance the `start` offset by the actual count
        # parsed off each page rather than a fixed stride — a fixed stride both
        # skips jobs (overshoot) and stops the loop early (empty overshoot page).
        start = 0
        for i in range(_MAX_PAGES):
            if loop.time() >= deadline or len(all_jobs) >= _MAX_JOBS:
                break
            url = ln.with_start(base, start)
            html, status = await self._fetch(page, url)
            if html is None:
                await on_event(
                    {
                        "type": "error",
                        "severity": "warning",
                        "portal": self.portal,
                        "message": f"listing start={start} HTTP {status}",
                    }
                )
                break
            cards = ln.parse_listing_cards(html)
            if not cards:
                break
            for job in cards:
                if job.id in seen:
                    continue
                seen.add(job.id)
                all_jobs.append(job)
            await on_event(
                {
                    "type": "progress",
                    "portal": self.portal,
                    "scraped": len(all_jobs),
                    "total": min(_MAX_JOBS, len(all_jobs) + 1),
                }
            )
            start += len(cards)
            if i + 1 < _MAX_PAGES:
                # "click" (1–3s) not "pagination" (3–7s): 5 pages of pagination
                # delay would consume most of the budget and starve enrichment.
                await asyncio.sleep(humanizer_delay("click"))

        all_jobs = all_jobs[:_MAX_JOBS]
        try:
            return await self._enrich_with_detail(page, all_jobs, deadline=deadline)
        except asyncio.CancelledError:
            logger.warning("linkedin: cancelled during enrichment; returning listing data")
            return all_jobs

    async def _scrape_voyager(
        self,
        page: Any,
        params: SearchParams,
        state: str,
        deadline: float,
        on_event: EventEmitter,
    ) -> list[JobListingDTO]:
        """Authenticated Voyager path: GraphQL search → cards, then per-job detail
        enrichment. Returns [] on any failure so the caller falls back to guest."""
        headers = lv.voyager_headers(state)
        if not headers:
            return []
        keywords = " ".join(params.role_keywords) or "engineer"
        location = " ".join(params.location) or "Indonesia"
        url = lv.voyager_search_url(keywords, location, start=0)
        try:
            resp = await page.request.get(url, headers=headers)
            body = await resp.text()
            if resp.status != 200:
                logger.warning(f"linkedin: Voyager search HTTP {resp.status}")
                return []
            jobs = vn.parse_voyager_search(body)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"linkedin: Voyager search failed: {e}")
            return []
        if not jobs:
            return []
        jobs = jobs[:_MAX_JOBS]
        await on_event(
            {
                "type": "progress",
                "portal": self.portal,
                "scraped": len(jobs),
                "total": len(jobs),
            }
        )
        await self._enrich_voyager(page, jobs, headers, deadline)
        return jobs

    async def _enrich_voyager(
        self,
        page: Any,
        jobs: list[JobListingDTO],
        headers: dict,
        deadline: float | None,
    ) -> None:
        """Fold each job's Voyager detail (full JD + structured fields) into the
        DTO in place; preserves the company name from the search card."""
        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(_ENRICH_CONCURRENCY)

        async def _one(idx: int, job: JobListingDTO) -> None:
            if deadline is not None and loop.time() >= deadline:
                return
            async with sem:
                # Re-check after acquiring the semaphore: a job can queue behind
                # the limiter long enough for the cooperative deadline to pass.
                if deadline is not None and loop.time() >= deadline:
                    return
                try:
                    resp = await page.request.get(lv.voyager_detail_url(job.id), headers=headers)
                    if resp.status != 200:
                        return
                    body = await resp.text()
                    parsed = vn.parse_voyager_detail(body, job.id)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"linkedin: Voyager detail {job.id} failed: {e}")
                    return
            if not parsed:
                return
            nj = parsed.get("detail")
            if nj is not None:
                nj.company = nj.company.model_copy(update={"name": job.company})
            jobs[idx] = job.model_copy(
                update={
                    "description": parsed.get("description") or job.description,
                    "seniority": parsed.get("seniority") or job.seniority,
                    "detail": nj,
                }
            )

        await asyncio.gather(*(_one(i, j) for i, j in enumerate(jobs)), return_exceptions=True)

    async def _fetch_job_detail(
        self, page: Any, job_id: str, *, deadline: float | None = None
    ) -> str | None:
        """Fetch a job's detail fragment, retrying on HTTP 429 with exponential
        backoff + jitter (LinkedIn's guest detail endpoint rate-limits bursts).
        Honors a Retry-After header when present. Each backoff is bounded by the
        cooperative deadline, so a throttled run still returns whatever enriched
        in time. Non-429 errors are not retried."""
        url = f"{_DETAIL_BASE}/{job_id}"
        loop = asyncio.get_event_loop()
        for attempt in range(_DETAIL_MAX_RETRIES + 1):
            try:
                resp = await page.request.get(url)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"linkedin: detail fetch {job_id} failed: {e}")
                return None
            if resp.status == 200:
                return await resp.text()
            if resp.status != 429 or attempt == _DETAIL_MAX_RETRIES:
                return None
            retry_after = _retry_after_seconds(resp.headers)
            delay = retry_after if retry_after is not None else _DETAIL_BACKOFF_BASE * (2**attempt)
            delay += random.uniform(0.0, 0.3)
            if deadline is not None and loop.time() + delay >= deadline:
                return None
            await asyncio.sleep(delay)
        return None

    async def _enrich_with_detail(
        self,
        page: Any,
        jobs: list[JobListingDTO],
        *,
        deadline: float | None = None,
        concurrency: int = _ENRICH_CONCURRENCY,
    ) -> list[JobListingDTO]:
        """Fetch each job's detail fragment (bounded concurrency, deadline-aware)
        and fold the rich NormalizedJob + plaintext + seniority into the DTO.
        Jobs not enriched in time keep their listing-derived data."""
        if not jobs:
            return jobs
        loop = asyncio.get_event_loop()
        out: list[JobListingDTO] = list(jobs)
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _enrich_one(idx: int, job: JobListingDTO) -> None:
            if deadline is not None and loop.time() >= deadline:
                return
            async with sem:
                if deadline is not None and loop.time() >= deadline:
                    return
                html = await self._fetch_job_detail(page, job.id, deadline=deadline)
            if not html:
                return
            parsed = ln.parse_job_detail(html, job.id)
            if parsed:
                out[idx] = job.model_copy(
                    update={
                        "description": parsed.get("description") or job.description,
                        "seniority": parsed.get("seniority") or job.seniority,
                        "detail": parsed.get("detail"),
                    }
                )

        await asyncio.gather(
            *(_enrich_one(i, j) for i, j in enumerate(jobs)),
            return_exceptions=True,
        )
        return out
