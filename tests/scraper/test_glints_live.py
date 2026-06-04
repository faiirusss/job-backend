import pytest

from app.schemas import SearchParams
from app.scrapers.base import browser_session
from app.scrapers.glints import GlintsScraper

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_glints_live_returns_jobs():
    scraper = GlintsScraper()
    params = SearchParams(role_keywords=["python"], location=["Jakarta"], work_type=["remote"])
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    async with browser_session("glints") as page:
        jobs = await scraper.scrape(page, params, on_event)

    assert isinstance(jobs, list)
    assert len(jobs) >= 1, f"expected ≥1 job from live Glints, got {len(jobs)}"
    j = jobs[0]
    assert j.portal == "glints"
    assert j.title
    assert j.company
    assert j.apply_url.startswith("http")
