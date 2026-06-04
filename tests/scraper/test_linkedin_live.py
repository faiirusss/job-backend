import pytest

from app.schemas import SearchParams
from app.scrapers.base import browser_session
from app.scrapers.linkedin import LinkedInScraper

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_linkedin_live_returns_jobs():
    scraper = LinkedInScraper()
    params = SearchParams(role_keywords=["python"], location=["Jakarta"])
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    async with browser_session("linkedin") as page:
        jobs = await scraper.scrape(page, params, on_event)

    assert isinstance(jobs, list)
    assert len(jobs) >= 1, f"expected ≥1 job from live LinkedIn, got {len(jobs)}"
    j = jobs[0]
    assert j.portal == "linkedin"
    assert j.title
    assert j.company
    assert j.apply_url.startswith("http")
    # at least one job should enrich with rich detail
    assert any(x.detail is not None and x.detail.description_html for x in jobs)
