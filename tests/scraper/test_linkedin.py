import pathlib

import pytest

from app.schemas import SearchParams
from app.scrapers.linkedin import LinkedInScraper

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
LISTING = (FIXTURES / "linkedin_listing.html").read_text(encoding="utf-8")
DETAIL = (FIXTURES / "linkedin_detail.html").read_text(encoding="utf-8")
VOY_SEARCH = (FIXTURES / "linkedin_voyager_search.json").read_text(encoding="utf-8")
VOY_DETAIL = (FIXTURES / "linkedin_voyager_detail.json").read_text(encoding="utf-8")


class _FakeResponse:
    def __init__(self, status: int, text: str, headers: dict | None = None):
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def text(self) -> str:
        return self._text


class _FakeRequest:
    def __init__(self, detail_calls: list[str]):
        self._detail_calls = detail_calls

    async def get(self, url: str):
        if "/jobPosting/" in url:
            self._detail_calls.append(url)
            return _FakeResponse(200, DETAIL)
        if "seeMoreJobPostings/search" in url:
            # Only page 1 (start=0) has results; later pages are empty → stop.
            if "start=0" in url:
                return _FakeResponse(200, LISTING)
            return _FakeResponse(200, "")
        return _FakeResponse(404, "")


class _FakePage:
    def __init__(self):
        self.detail_calls: list[str] = []
        self.request = _FakeRequest(self.detail_calls)


def test_build_search_url_encodes_keywords_and_location():
    url = LinkedInScraper().build_search_url(
        SearchParams(role_keywords=["python", "backend"], location=["Jakarta"])
    )
    assert url.startswith("https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?")
    assert "keywords=python+backend" in url
    assert "location=Jakarta" in url


@pytest.mark.asyncio
async def test_scrape_returns_enriched_jobs():
    scraper = LinkedInScraper()
    page = _FakePage()
    params = SearchParams(role_keywords=["python"], location=["Jakarta"])
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    jobs = await scraper.scrape(page, params, on_event)

    assert len(jobs) == 2
    enriched = next(j for j in jobs if j.id == "3901234567")
    assert enriched.detail is not None
    assert enriched.detail.job_type_label == "Penuh Waktu"
    assert enriched.seniority == "senior"  # overridden from detail criteria
    assert "backend engineer" in enriched.description.lower()
    assert any(e["type"] == "progress" for e in events)
    assert len(page.detail_calls) == 2  # both jobs enriched


class _FailingRequest:
    async def get(self, url: str):
        if "seeMoreJobPostings/search" in url:
            return _FakeResponse(403, "")
        return _FakeResponse(404, "")


class _FailingPage:
    def __init__(self):
        self.request = _FailingRequest()


@pytest.mark.asyncio
async def test_scrape_listing_failure_returns_empty_and_warns():
    scraper = LinkedInScraper()
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    jobs = await scraper.scrape(_FailingPage(), SearchParams(role_keywords=["python"]), on_event)

    assert jobs == []
    assert any(e["type"] == "error" and e.get("severity") == "warning" for e in events)


class _ListingOnlyRequest:
    async def get(self, url: str):
        if "/jobPosting/" in url:
            return _FakeResponse(500, "")  # detail fetch fails
        if "seeMoreJobPostings/search" in url and "start=0" in url:
            return _FakeResponse(200, LISTING)
        return _FakeResponse(200, "")


class _ListingOnlyPage:
    def __init__(self):
        self.request = _ListingOnlyRequest()


@pytest.mark.asyncio
async def test_scrape_preserves_listing_when_detail_fails():
    scraper = LinkedInScraper()

    async def on_event(ev):
        pass

    jobs = await scraper.scrape(
        _ListingOnlyPage(), SearchParams(role_keywords=["python"]), on_event
    )

    assert len(jobs) == 2  # listing cards preserved even though detail fetch failed
    assert all(j.detail is None for j in jobs)
    assert all(j.title for j in jobs)


# Page 2: one DUPLICATE of a page-1 job (3901234567) + one NEW job. The guest
# endpoint returns fewer cards per call than the nominal page size, so the next
# page must be requested at start = (cards seen so far), i.e. start=2 here — NOT
# a fixed stride of 25 (which would skip jobs and stop the loop early).
_PAGE2 = (
    '<div data-entity-urn="urn:li:jobPosting:3901234567">'
    '<a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/dup"><span>x</span></a>'
    '<h3 class="base-search-card__title">Senior Backend Engineer</h3></div>'
    '<div data-entity-urn="urn:li:jobPosting:5550001111">'
    '<a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/new"><span>x</span></a>'
    '<h3 class="base-search-card__title">Data Engineer</h3></div>'
)


class _PagedRequest:
    """Page 1 (start=0) returns 2 cards; the next page is only reachable at
    start=2 (the actual card count), exercising count-based pagination."""

    def __init__(self):
        self.listing_starts: list[str] = []

    async def get(self, url: str):
        if "/jobPosting/" in url:
            return _FakeResponse(200, DETAIL)
        if "seeMoreJobPostings/search" in url:
            start = url.split("start=", 1)[1].split("&", 1)[0]
            self.listing_starts.append(start)
            if start == "0":
                return _FakeResponse(200, LISTING)  # ids 3901234567, 3907654321
            if start == "2":
                return _FakeResponse(200, _PAGE2)  # dup 3901234567 + new 5550001111
            return _FakeResponse(200, "")  # start=4 → empty → stop
        return _FakeResponse(404, "")


class _PagedPage:
    def __init__(self):
        self.request = _PagedRequest()


@pytest.mark.asyncio
async def test_scrape_paginates_and_dedupes_across_pages():
    scraper = LinkedInScraper()
    page = _PagedPage()

    async def on_event(ev):
        pass

    jobs = await scraper.scrape(page, SearchParams(role_keywords=["python"]), on_event)

    ids = sorted(j.id for j in jobs)
    # 2 from page 1 + 1 new from page 2; the page-2 duplicate is deduped away.
    assert ids == ["3901234567", "3907654321", "5550001111"]
    # Pagination advances by the actual card count (0 → 2), not a fixed stride.
    assert page.request.listing_starts[:2] == ["0", "2"]


class _RateLimitedRequest:
    """Returns 429 for the first two attempts on each detail job, then 200.
    Tracks attempts per job id so the test can assert retry happened."""

    def __init__(self):
        self.detail_attempts: dict[str, int] = {}

    async def get(self, url: str):
        if "/jobPosting/" in url:
            jid = url.rsplit("/", 1)[-1]
            n = self.detail_attempts.get(jid, 0)
            self.detail_attempts[jid] = n + 1
            if n < 2:
                return _FakeResponse(429, "", headers={"retry-after": "0"})
            return _FakeResponse(200, DETAIL)
        if "seeMoreJobPostings/search" in url and "start=0" in url:
            return _FakeResponse(200, LISTING)
        return _FakeResponse(200, "")


class _RateLimitedPage:
    def __init__(self):
        self.request = _RateLimitedRequest()


class _ServerErrorRequest:
    """Detail endpoint always 500 (a non-429 error → must NOT be retried)."""

    def __init__(self):
        self.detail_attempts: dict[str, int] = {}

    async def get(self, url: str):
        if "/jobPosting/" in url:
            jid = url.rsplit("/", 1)[-1]
            self.detail_attempts[jid] = self.detail_attempts.get(jid, 0) + 1
            return _FakeResponse(500, "")
        if "seeMoreJobPostings/search" in url and "start=0" in url:
            return _FakeResponse(200, LISTING)
        return _FakeResponse(200, "")


class _ServerErrorPage:
    def __init__(self):
        self.request = _ServerErrorRequest()


@pytest.mark.asyncio
async def test_scrape_retries_detail_on_429(monkeypatch):
    # No real backoff sleeps: we test the retry LOGIC, not the delay duration.
    async def _nosleep(*a, **k):
        return None

    monkeypatch.setattr("app.scrapers.linkedin.asyncio.sleep", _nosleep)
    scraper = LinkedInScraper()
    page = _RateLimitedPage()

    async def on_event(ev):
        pass

    jobs = await scraper.scrape(page, SearchParams(role_keywords=["python"]), on_event)

    enriched = [j for j in jobs if j.detail is not None]
    assert len(enriched) == 2  # both jobs enriched after retrying past the 429s
    # each job: 2x429 + 1x200 = 3 attempts
    assert page.request.detail_attempts["3901234567"] == 3
    assert page.request.detail_attempts["3907654321"] == 3


@pytest.mark.asyncio
async def test_scrape_does_not_retry_non_429_errors(monkeypatch):
    async def _nosleep(*a, **k):
        return None

    monkeypatch.setattr("app.scrapers.linkedin.asyncio.sleep", _nosleep)
    scraper = LinkedInScraper()
    page = _ServerErrorPage()

    async def on_event(ev):
        pass

    jobs = await scraper.scrape(page, SearchParams(role_keywords=["python"]), on_event)

    assert all(j.detail is None for j in jobs)  # 500 → no enrichment
    # 500 is not retried: exactly one attempt per job
    assert page.request.detail_attempts["3901234567"] == 1


class _VoyagerRequest:
    """Serves the Voyager graphql search + per-job detail from fixtures."""

    async def get(self, url, headers=None):
        if "/voyager/api/graphql" in url:
            return _FakeResponse(200, VOY_SEARCH)
        if "/voyager/api/jobs/jobPostings/" in url:
            return _FakeResponse(200, VOY_DETAIL)
        return _FakeResponse(404, "")


class _VoyagerPage:
    def __init__(self):
        self.request = _VoyagerRequest()


@pytest.mark.asyncio
async def test_scrape_uses_voyager_when_session_present(monkeypatch):
    from app.scrapers import linkedin as li

    monkeypatch.setattr(li.linkedin_auth, "storage_state_path", lambda: "/tmp/s.json")
    monkeypatch.setattr(li.lv, "voyager_headers", lambda _p: {"csrf-token": "x"})
    monkeypatch.setattr("app.scrapers.linkedin.humanizer_delay", lambda _k: 0)

    scraper = LinkedInScraper()

    async def on_event(ev):
        pass

    jobs = await scraper.scrape(
        _VoyagerPage(), SearchParams(role_keywords=["software engineer"]), on_event
    )

    # 3 cards from the Voyager search fixture, enriched with the detail JD.
    assert len(jobs) == 3
    j = next(x for x in jobs if x.id == "4277612327")
    assert j.company == "PT IKONSULTAN INOVATAMA"
    # at least one job carries the real Voyager description folded in
    assert any(x.description.startswith("Experience IT Developer") for x in jobs)
    assert any(x.detail is not None for x in jobs)
    # the detail's company name is backfilled from the listing card
    assert j.detail is not None and j.detail.company.name == "PT IKONSULTAN INOVATAMA"


class _PagedVoyagerRequest:
    def __init__(self):
        self.starts: list[str] = []

    async def get(self, url, headers=None):
        if "/voyager/api/graphql" in url:
            start = url.split("start:", 1)[1].split(")", 1)[0]
            self.starts.append(start)
            if start == "0":
                return _FakeResponse(200, VOY_SEARCH)
            if start == "25":
                body = (
                    VOY_SEARCH.replace("4277612327", "5277612327")
                    .replace("4418621292", "5418621292")
                    .replace("4332554777", "5332554777")
                )
                return _FakeResponse(200, body)
            return _FakeResponse(200, '{"included":[]}')
        if "/voyager/api/jobs/jobPostings/" in url:
            return _FakeResponse(200, VOY_DETAIL)
        return _FakeResponse(404, "")


class _PagedVoyagerPage:
    def __init__(self):
        self.request = _PagedVoyagerRequest()


@pytest.mark.asyncio
async def test_voyager_scrape_paginates_search_results(monkeypatch):
    from app.scrapers import linkedin as li

    monkeypatch.setattr(li.linkedin_auth, "storage_state_path", lambda: "/tmp/s.json")
    monkeypatch.setattr(li.lv, "voyager_headers", lambda _p: {"csrf-token": "x"})
    monkeypatch.setattr("app.scrapers.linkedin.humanizer_delay", lambda _k: 0)

    scraper = LinkedInScraper()
    page = _PagedVoyagerPage()

    async def on_event(ev):
        pass

    jobs = await scraper.scrape(
        page, SearchParams(role_keywords=["software engineer"]), on_event
    )

    assert len(jobs) == 6
    assert page.request.starts == ["0", "25", "50"]


@pytest.mark.asyncio
async def test_scrape_falls_back_to_guest_without_session(monkeypatch):
    from app.scrapers import linkedin as li

    monkeypatch.setattr(li.linkedin_auth, "storage_state_path", lambda: None)
    scraper = LinkedInScraper()

    async def on_event(ev):
        pass

    # _FakePage serves the guest listing/detail fixtures (2 jobs).
    jobs = await scraper.scrape(_FakePage(), SearchParams(role_keywords=["python"]), on_event)
    assert len(jobs) == 2  # guest path
