import asyncio
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import JobListing
from app.schemas import NormalizedJob
from app.scrapers import (
    linkedin_auth,
    linkedin_normalize as ln,
    linkedin_voyager as lv,
    linkedin_voyager_normalize as vn,
)
from app.scrapers.base import browser_session
from app.scrapers.linkedin import LinkedInScraper

_DETAIL_TIMEOUT_SECONDS = 25
_GUEST_DETAIL_DEADLINE_SECONDS = 18


async def ensure_job_detail(session: AsyncSession, job: JobListing) -> bool:
    """Best-effort lazy enrichment for listing-only LinkedIn jobs.

    LinkedIn search cards often do not include the full JD. Search should still
    show those cards quickly, then this service fills the rich detail when a user
    opens the job or asks for analysis. Returns True only when the DB row changed.
    """
    if not _needs_linkedin_detail(job):
        return False

    parsed = await fetch_linkedin_detail(str(job.external_id), company=job.company)
    if not parsed:
        return False

    changed = _apply_linkedin_detail(job, parsed)
    if changed:
        await session.flush()
    return changed


async def fetch_linkedin_detail(job_id: str, *, company: str = "") -> dict[str, Any] | None:
    """Fetch and normalize one LinkedIn job detail payload.

    Authenticated Voyager is preferred when a saved session exists; the anonymous
    guest detail endpoint is the fallback. Never raises to callers: detail fetch
    failure should not make opening a job fail.
    """
    storage_state = linkedin_auth.storage_state_path()
    try:
        return await asyncio.wait_for(
            _fetch_linkedin_detail_with_browser(job_id, company=company, storage_state=storage_state),
            timeout=_DETAIL_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning(f"linkedin: lazy detail fetch timed out for {job_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"linkedin: lazy detail fetch failed for {job_id}: {e}")
    return None


async def _fetch_linkedin_detail_with_browser(
    job_id: str,
    *,
    company: str,
    storage_state: str | None,
) -> dict[str, Any] | None:
    async with browser_session("linkedin", storage_state=storage_state) as page:
        if storage_state:
            parsed = await _fetch_voyager_detail(page, job_id, storage_state, company=company)
            if parsed:
                return parsed

        scraper = LinkedInScraper()
        loop = asyncio.get_event_loop()
        html = await scraper._fetch_job_detail(
            page,
            job_id,
            deadline=loop.time() + _GUEST_DETAIL_DEADLINE_SECONDS,
        )
        if not html:
            return None
        return ln.parse_job_detail(html, job_id) or None


async def _fetch_voyager_detail(
    page: Any,
    job_id: str,
    storage_state: str,
    *,
    company: str,
) -> dict[str, Any] | None:
    headers = lv.voyager_headers(storage_state)
    if not headers:
        return None
    try:
        resp = await page.request.get(lv.voyager_detail_url(job_id), headers=headers)
        if resp.status != 200:
            return None
        parsed = vn.parse_voyager_detail(await resp.text(), job_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"linkedin: lazy Voyager detail {job_id} failed: {e}")
        return None
    if not parsed:
        return None
    detail = parsed.get("detail")
    if isinstance(detail, NormalizedJob) and company:
        detail.company = detail.company.model_copy(update={"name": company})
    return parsed


def _needs_linkedin_detail(job: JobListing) -> bool:
    if job.portal != "linkedin":
        return False
    return not (job.description and job.detail_json)


def _apply_linkedin_detail(job: JobListing, parsed: dict[str, Any]) -> bool:
    changed = False
    description = str(parsed.get("description") or "").strip()
    if description and description != (job.description or ""):
        job.description = description
        job.embedding = None
        job.responsibilities = None
        job.mandatory_requirements = None
        job.nice_to_have_requirements = None
        job.skills_tags = None
        job.benefits = None
        changed = True

    seniority = str(parsed.get("seniority") or "").strip()
    if seniority and seniority != (job.seniority or ""):
        job.seniority = seniority
        changed = True

    detail = parsed.get("detail")
    if isinstance(detail, NormalizedJob):
        detail_json = detail.model_dump(mode="json")
        if detail_json != (job.detail_json or None):
            job.detail_json = detail_json
            changed = True

    if changed:
        job.scraped_at = datetime.now(UTC)
    return changed
