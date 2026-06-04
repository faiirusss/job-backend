import json
import pathlib
from pathlib import Path
from urllib.parse import quote_plus

import pytest

from app.schemas import SearchParams
from app.scrapers.glints import GlintsScraper, _parse_job_detail

FIXTURE = Path(__file__).parent.parent / "fixtures" / "glints_response.json"

_NEXT = pathlib.Path(__file__).parents[1] / "fixtures" / "glints_next_data.json"


def test_parse_job_detail_returns_detail_and_plaintext():
    parsed = _parse_job_detail(json.loads(_NEXT.read_text()))
    assert parsed["detail"].title == "Fullstack Engineer"
    # description is plain text (no tags) for embedding
    assert "<" not in parsed["description"]
    # Exact structured arrays from the fixture (locks content, not just length)
    assert parsed["skills_tags"] == ["Node.js", "Laravel", "JavaScript"]
    assert "Career Path" in parsed["benefits"]
    # Description content is preserved verbatim (plain text, block-joined)
    assert (
        "Provide solution and execute implementation for website development"
        in parsed["description"]
    )
    assert "Job Descriptions" in parsed["description"]


def test_parse_job_detail_empty_on_unknown_shape():
    assert _parse_job_detail({"foo": "bar"}) == {}


def test_build_search_url_encodes_keywords():
    s = GlintsScraper()
    params = SearchParams(
        role_keywords=["Node.js", "fullstack"], location=["Jakarta"], work_type=["remote"]
    )
    url = s.build_search_url(params)
    assert "glints.com" in url
    assert quote_plus("Node.js") in url or "Node.js" in url


@pytest.mark.asyncio
async def test_scrape_parses_intercepted_json():
    """Scraper should produce JobListingDTO entries from intercepted JSON."""
    s = GlintsScraper()
    raw = json.loads(FIXTURE.read_text())
    jobs = s._parse_api_response(raw)
    assert len(jobs) == 2
    j = jobs[0]
    assert j.portal == "glints"
    assert j.title == "Backend Engineer (Node.js)"
    assert j.company == "Tokopedia"
    assert j.salary_min == 15000000
    assert j.work_type == "remote"
    assert j.apply_url.startswith("https://")
    assert j.match_score is None
    assert j.company_logo_bg.startswith("#")


@pytest.mark.asyncio
async def test_scrape_does_not_emit_partial_result(monkeypatch):
    """Per the new contract, scrapers emit progress only — search_service
    owns partial_result emission."""
    from app.schemas import SearchParams
    from app.scrapers.glints import GlintsScraper

    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    scraper = GlintsScraper()
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    # Stub the page object — we will not navigate; we only exercise the parser path.
    class FakePage:
        def on(self, *_args, **_kwargs):
            pass

        async def goto(self, *_args, **_kwargs):
            pass

        async def content(self):
            return "<html></html>"

    page = FakePage()
    params = SearchParams(role_keywords=["python"], location=["Jakarta"], work_type=["remote"])
    await scraper.scrape(page, params, on_event)

    types = {e.get("type") for e in events}
    assert "partial_result" not in types, f"unexpected partial_result events: {events}"


def test_parse_api_response_handles_alternate_shape():
    """The parser should handle Glints's alternate key naming (jobTitle,
    companyName, locationName, salary.min/max, jobDescription, jobRequirement, url)."""
    import json
    from pathlib import Path

    from app.scrapers.glints import GlintsScraper

    s = GlintsScraper()
    raw = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "glints_response_v2.json").read_text()
    )
    jobs = s._parse_api_response(raw)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.title == "Frontend Engineer (React)"
    assert j.company == "Tokopedia"
    assert j.location == "Jakarta"
    assert j.salary_min == 12000000
    assert j.salary_max == 18000000
    assert j.work_type == "remote"
    assert "React" in j.description
    assert j.requirements
    assert j.apply_url.startswith("https://glints.com")


def test_parse_api_response_handles_graphql_searchjobsv3_shape():
    """Live Glints serves jobs via GraphQL: {"data": {"searchJobsV3":
    {"jobsInPage": [...]}}}. The parser must unwrap jobsInPage and read the
    `salaries[].min/maxAmount` salary shape. Regression for the 0-results bug."""
    import json
    from pathlib import Path

    from app.scrapers.glints import GlintsScraper

    s = GlintsScraper()
    raw = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "glints_response_graphql.json").read_text()
    )
    jobs = s._parse_api_response(raw)
    assert len(jobs) == 2, "GraphQL searchJobsV3.jobsInPage must be unwrapped"
    by_title = {j.title: j for j in jobs}
    assert "Backend Engineer" in by_title
    react = by_title["Senior Frontend Engineer (React)"]
    assert react.company == "Tokopedia"
    assert react.location == "Jakarta"
    assert react.work_type == "remote"
    assert react.salary_min == 5000000
    assert react.salary_max == 7000000


def test_parse_api_response_skips_items_without_id_or_title():
    """Items lacking both id and title are skipped (avoids 'Unknown role' rows)."""
    from app.scrapers.glints import GlintsScraper

    s = GlintsScraper()
    payload = {
        "data": [
            {"id": "ok-1", "title": "Engineer", "company": {"name": "X"}},
            {"company": {"name": "Y"}, "description": "no id no title"},
            {"id": "", "title": "", "company": {"name": "Z"}},
        ]
    }
    jobs = s._parse_api_response(payload)
    assert len(jobs) == 1
    assert jobs[0].title == "Engineer"


def test_parse_html_salary_handles_common_formats():
    from app.scrapers.glints import _parse_html_salary

    assert _parse_html_salary("Rp 12 Jt - 18 Jt") == (12_000_000, 18_000_000)
    assert _parse_html_salary("IDR 5.000.000 - 8.000.000") == (5_000_000, 8_000_000)
    assert _parse_html_salary("") == (0, 0)
    assert _parse_html_salary("Negotiable") == (0, 0)


def test_with_page_param_replaces_existing_page_param():
    from app.scrapers.glints import _with_page_param

    assert (
        _with_page_param("https://example.com/api?keyword=x", 2)
        == "https://example.com/api?keyword=x&page=2"
    )
    assert (
        _with_page_param("https://example.com/api?keyword=x&page=1", 3)
        == "https://example.com/api?keyword=x&page=3"
    )


@pytest.mark.asyncio
async def test_scrape_paginates_graphql_via_post_replay(monkeypatch):
    """Glints serves jobs from a GraphQL POST. Pagination must replay the
    captured POST body with an incremented `page` variable (not a GET ?page=)."""
    from app.schemas import SearchParams
    from app.scrapers.glints import GlintsScraper

    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    s = GlintsScraper()
    events: list[dict] = []
    posted_bodies: list[str] = []

    async def on_event(ev):
        events.append(ev)

    base_body = json.dumps(
        {
            "operationName": "searchJobsV3",
            "variables": {
                "data": {"SearchTerm": "x", "CountryCode": "ID", "pageSize": 30, "page": 1}
            },
            "query": "query searchJobsV3 { searchJobsV3 { jobsInPage { id } } }",
        }
    )

    def payload_for(page_num: int) -> dict:
        if page_num >= 4:
            return {"data": {"searchJobsV3": {"jobsInPage": []}}}
        return {
            "data": {
                "searchJobsV3": {
                    "jobsInPage": [
                        {"id": f"p{page_num}-{i}", "title": f"Job {i}", "company": {"name": "Co"}}
                        for i in range(10)
                    ]
                }
            }
        }

    class FakeReq:
        method = "POST"
        post_data = base_body

    class FakeResp:
        def __init__(self, url, body):
            self.url = url
            self.headers = {"content-type": "application/json"}
            self._body = body
            self.request = FakeReq()

        async def json(self):
            return self._body

    class FakePage:
        def __init__(self):
            self._handlers: list = []

        def on(self, _event, cb):
            self._handlers.append(cb)

        async def goto(self, *_args, **_kwargs):
            await self._handlers[0](
                FakeResp(
                    "https://glints.com/api/v2-alc/graphql?op=searchJobsV3",
                    payload_for(1),
                )
            )

        async def content(self):
            return ""

        async def evaluate(self, _script, arg):
            if isinstance(arg, str):
                return ""  # detail fetch — empty HTML so _fetch_job_detail returns {}
            assert isinstance(arg, list), "expected [url, body, headers] for POST replay"
            _url, body, _headers = arg
            posted_bodies.append(body)
            page_num = json.loads(body)["variables"]["data"]["page"]
            return payload_for(page_num)

    params = SearchParams(role_keywords=["x"], location=["Jakarta"], work_type=["remote"])
    jobs = await s.scrape(FakePage(), params, on_event)

    # p1(10) + p2(10) + p3(10) = 30 unique; p4 empty stops the loop.
    assert len(jobs) == 30
    assert [json.loads(b)["variables"]["data"]["page"] for b in posted_bodies] == [2, 3, 4]


@pytest.mark.asyncio
async def test_scrape_paginates_up_to_max_pages(monkeypatch):
    """When the first XHR is captured and the URL is known, scrape should
    request additional pages via page.evaluate and merge unique jobs."""
    from app.schemas import SearchParams
    from app.scrapers.glints import _MAX_PAGES, GlintsScraper

    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    s = GlintsScraper()
    events: list[dict] = []
    pages_fetched: list[str] = []

    async def on_event(ev):
        events.append(ev)

    class FakeResp:
        def __init__(self, url, body):
            self.url = url
            self.headers = {"content-type": "application/json"}
            self._body = body

        async def json(self):
            return self._body

    class FakePage:
        _handlers: list = []

        def on(self, _event, cb):
            self._handlers.append(cb)

        async def goto(self, *_args, **_kwargs):
            # Simulate the first-page XHR firing once after navigation.
            await self._handlers[0](
                FakeResp(
                    "https://glints.com/api/v2/job-listings?keyword=x",
                    {
                        "data": [
                            {"id": f"p1-{i}", "title": f"Job {i}", "company": {"name": "Co"}}
                            for i in range(10)
                        ]
                    },
                )
            )

        async def content(self):
            return ""

        async def evaluate(self, _script, url):
            if "opportunities/jobs/" in str(url):
                return ""  # detail fetch — empty HTML
            pages_fetched.append(url)
            page_num = int(url.split("page=")[-1])
            if page_num >= 4:
                return {"data": []}
            return {
                "data": [
                    {"id": f"p{page_num}-{i}", "title": f"Job {i}", "company": {"name": "Co"}}
                    for i in range(10)
                ]
            }

    params = SearchParams(role_keywords=["x"], location=["Jakarta"], work_type=["remote"])
    jobs = await s.scrape(FakePage(), params, on_event)

    # 10 from p1 + 10 from p2 + 10 from p3 = 30 unique. p4 was empty so loop stopped.
    assert len(jobs) == 30
    # Should have fetched at most _MAX_PAGES-1 additional URLs (pages 2,3,4)
    assert all("page=" in u for u in pages_fetched)
    assert len(pages_fetched) <= _MAX_PAGES - 1


@pytest.mark.asyncio
async def test_scrape_keeps_payload_when_navigation_times_out(monkeypatch):
    """Regression: page.goto can time out because DOMContentLoaded never fires on
    Glints' Cloudflare-fronted Next.js page (esp. headless over WSL2's IPv6 route),
    yet the searchJobsV3 XHR — the data we actually consume — has already been
    intercepted. A goto timeout must NOT discard that captured payload."""
    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    s = GlintsScraper()
    raw = json.loads(FIXTURE.read_text())  # 2 jobs
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    class FakeReq:
        post_data = None

        async def all_headers(self):
            return {}

    class FakeResp:
        url = "https://glints.com/api/v2-alc/graphql?op=searchJobsV3"
        headers = {"content-type": "application/json"}
        request = FakeReq()

        async def json(self):
            return raw

    class FakePage:
        def __init__(self):
            self._handler = None

        def on(self, _event, cb):
            self._handler = cb

        async def goto(self, *_args, **_kwargs):
            # The XHR fires during navigation; then DOMContentLoaded never settles.
            if self._handler:
                await self._handler(FakeResp())
            raise TimeoutError("Page.goto: Timeout 30000ms exceeded.")

        async def evaluate(self, *_args, **_kwargs):
            return {}

        async def content(self):
            return "<html></html>"

    params = SearchParams(role_keywords=["python"], location=["Jakarta"], work_type=["remote"])
    jobs = await s.scrape(FakePage(), params, on_event)

    assert len(jobs) == 2, f"navigation timeout discarded the captured payload: {jobs}"
    assert {j.portal for j in jobs} == {"glints"}


def test_parse_job_detail_extracts_structured_arrays_and_full_description():
    """_parse_job_detail delegates to extract_glints_job (reads
    props.pageProps.initialData.data). Returns description (plain text), skills_tags,
    benefits (flat titles), and the full NormalizedJob as `detail`."""
    parsed = _parse_job_detail(json.loads(_NEXT.read_text()))
    # New shape: the four keys from the rewritten implementation.
    assert "detail" in parsed
    assert "description" in parsed
    assert "skills_tags" in parsed
    assert "benefits" in parsed
    # No regex-split prose sections in the returned dict.
    assert "responsibilities" not in parsed
    assert "mandatory_requirements" not in parsed
    # description is plain text (no HTML tags).
    desc = parsed["description"]
    assert "<" not in desc
    # Skills and benefits are populated from the fixture.
    assert len(parsed["skills_tags"]) > 0
    assert len(parsed["benefits"]) > 0


def test_parse_job_detail_preserves_indonesian_description_block():
    """A payload that lacks the props.pageProps.initialData.data path returns {}
    (the old props.pageProps.job path is no longer supported)."""
    payload = {
        "props": {
            "pageProps": {
                "job": {
                    "descriptionHtml": "Membangun API dengan FastAPI",
                    "skills": [{"skill": {"name": "Python"}}],
                    "benefits": [{"title": "Asuransi Kesehatan"}],
                }
            }
        }
    }
    # Old path is not read; extract_glints_job returns None -> empty dict.
    result = _parse_job_detail(payload)
    assert result == {}


@pytest.mark.asyncio
async def test_enrichment_respects_deadline_and_returns_all_jobs(monkeypatch):
    """When the enrichment budget is exhausted, _enrich_with_detail must still
    return every job (with unenriched ones falling back to listing data). Regression
    for the bug where a 45 s portal timeout firing mid-enrichment caused the
    orchestrator to discard all 100 already-scraped jobs.
    """
    import asyncio as _asyncio

    from app.schemas import JobListingDTO
    from app.scrapers.glints import GlintsScraper

    # No humanizer sleeps in unit tests — we control timing via the page stub.
    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    scraper = GlintsScraper()

    def _job(idx: int) -> JobListingDTO:
        return JobListingDTO(
            id=f"job-{idx}",
            portal="glints",
            title=f"Job {idx}",
            company="Co",
            company_logo_bg="#000000",
            location="Indonesia",
            work_type="remote",
            seniority="mid",
            salary_min=0,
            salary_max=0,
            posted_date="2026-01-01",
            posted_label="recent",
            apply_url=f"https://glints.com/id/opportunities/jobs/job-{idx}",
            match_score=None,
            cosine=0.0,
            llm_score=0,
            matched_skills=[],
            missing_skills=[],
            summary_id="",
            summary_en="",
            description=f"listing desc {idx}",
            requirements="",
        )

    jobs = [_job(i) for i in range(20)]

    class SlowPage:
        async def evaluate(self, _script, _arg):
            # Each detail fetch takes "forever" in test time — guarantees the
            # deadline trips before any single fetch resolves.
            await _asyncio.sleep(10)
            return "<html></html>"

    loop = _asyncio.get_event_loop()
    deadline = loop.time() + 0.05  # 50 ms — practically immediate

    out = await scraper._enrich_with_detail(SlowPage(), jobs, deadline=deadline)

    assert len(out) == len(jobs), (
        "every job must be preserved even when enrichment runs out of time"
    )
    # Unenriched jobs keep their original listing-derived description.
    assert {j.description for j in out} == {f"listing desc {i}" for i in range(20)}


@pytest.mark.asyncio
async def test_enrichment_runs_concurrently(monkeypatch):
    """Enrichment must dispatch detail fetches in parallel. With 30 jobs and a
    100ms fetch latency, serial execution would take ~3s; bounded concurrency
    of ~6 finishes in well under 1s.
    """
    import asyncio as _asyncio
    import time as _time

    from app.schemas import JobListingDTO
    from app.scrapers.glints import GlintsScraper

    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    scraper = GlintsScraper()
    jobs = [
        JobListingDTO(
            id=f"j{i}",
            portal="glints",
            title=f"T{i}",
            company="C",
            company_logo_bg="#000000",
            location="ID",
            work_type="remote",
            seniority="mid",
            salary_min=0,
            salary_max=0,
            posted_date="2026-01-01",
            posted_label="recent",
            apply_url=f"https://glints.com/id/opportunities/jobs/j{i}",
            match_score=None,
            cosine=0.0,
            llm_score=0,
            matched_skills=[],
            missing_skills=[],
            summary_id="",
            summary_en="",
            description="",
            requirements="",
        )
        for i in range(30)
    ]

    class FixedLatencyPage:
        async def evaluate(self, _script, _arg):
            await _asyncio.sleep(0.1)  # 100 ms per fetch
            return ""  # empty HTML → _fetch_job_detail returns {}, listing kept as-is

    t0 = _time.monotonic()
    out = await scraper._enrich_with_detail(FixedLatencyPage(), jobs)
    elapsed = _time.monotonic() - t0

    assert len(out) == 30
    # Serial would be ~3.0s; bounded-concurrency target is <1.0s.
    assert elapsed < 1.5, f"enrichment took {elapsed:.2f}s — not running concurrently"


@pytest.mark.asyncio
async def test_scrape_returns_all_jobs_when_enrichment_budget_zero(monkeypatch):
    """End-to-end: even if the enrichment deadline is past, scrape() returns
    the full listing-scrape result. Plus the orchestrator no longer needs to
    catch a TimeoutError to surface results.
    """
    import asyncio as _asyncio

    from app.config import settings as _settings
    from app.schemas import SearchParams
    from app.scrapers.glints import GlintsScraper

    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)
    # Force scrape() to compute a deadline that's already passed: zero budget.
    monkeypatch.setattr(_settings, "scraper_timeout_seconds", 0)

    scraper = GlintsScraper()
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    class FakeResp:
        def __init__(self, url, body):
            self.url = url
            self.headers = {"content-type": "application/json"}
            self._body = body

            class _R:
                method = "POST"
                post_data = None

                async def all_headers(self):
                    return {}

            self.request = _R()

        async def json(self):
            return self._body

    class FakePage:
        def __init__(self):
            self._handlers: list = []

        def on(self, _event, cb):
            self._handlers.append(cb)

        async def goto(self, *_args, **_kwargs):
            await self._handlers[0](
                FakeResp(
                    "https://glints.com/api/v2/job-listings?keyword=x",
                    {
                        "data": [
                            {"id": f"j{i}", "title": f"T{i}", "company": {"name": "Co"}}
                            for i in range(15)
                        ]
                    },
                )
            )

        async def content(self):
            return ""

        async def evaluate(self, _script, _arg):
            # Detail fetches would be slow, but the zero-budget deadline must
            # short-circuit them and still return all 15 jobs intact.
            await _asyncio.sleep(5)
            return ""

    params = SearchParams(role_keywords=["x"], location=["Jakarta"], work_type=["remote"])
    jobs = await scraper.scrape(FakePage(), params, on_event)
    assert len(jobs) == 15, "listing-only results must survive an exhausted enrichment budget"
