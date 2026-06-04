import json

import pytest

from app.config import settings
from app.schemas import SearchParams
from app.scrapers import glints_auth
from app.scrapers.glints import GlintsScraper


# --------------------------------------------------------------------------- #
# glints_auth.storage_state_path
# --------------------------------------------------------------------------- #
def test_storage_state_path_returns_path_for_valid_file(tmp_path, monkeypatch):
    p = tmp_path / "glints_state.json"
    p.write_text(json.dumps({"cookies": [], "origins": []}))
    monkeypatch.setattr(settings, "glints_storage_state_path", str(p))
    assert glints_auth.storage_state_path() == str(p)


def test_storage_state_path_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "glints_storage_state_path", str(tmp_path / "nope.json"))
    assert glints_auth.storage_state_path() is None


def test_storage_state_path_returns_none_for_invalid_json(tmp_path, monkeypatch):
    p = tmp_path / "glints_state.json"
    p.write_text("{not json")
    monkeypatch.setattr(settings, "glints_storage_state_path", str(p))
    assert glints_auth.storage_state_path() is None


@pytest.mark.asyncio
async def test_refresh_session_noop_without_credentials(monkeypatch):
    monkeypatch.setattr(settings, "glints_email", "")
    monkeypatch.setattr(settings, "glints_password", "")
    assert await glints_auth.refresh_session() is None


@pytest.mark.asyncio
async def test_ensure_session_returns_existing_without_login(tmp_path, monkeypatch):
    """A valid session file is reused as-is — no headless login is attempted."""
    p = tmp_path / "glints_state.json"
    p.write_text(json.dumps({"cookies": [], "origins": []}))
    monkeypatch.setattr(settings, "glints_storage_state_path", str(p))

    called = {"login": False}

    async def _no_login():
        called["login"] = True
        return None

    monkeypatch.setattr(glints_auth, "_login_and_save", _no_login)
    assert await glints_auth.ensure_session() == str(p)
    assert called["login"] is False


@pytest.mark.asyncio
async def test_ensure_session_auto_seeds_when_missing(tmp_path, monkeypatch):
    """Missing file + credentials -> ensure_session performs the headless login
    and returns the freshly written, validated path."""
    p = tmp_path / "glints_state.json"
    monkeypatch.setattr(settings, "glints_storage_state_path", str(p))
    monkeypatch.setattr(settings, "glints_email", "bot@example.com")
    monkeypatch.setattr(settings, "glints_password", "pw")

    async def _fake_login():
        p.write_text(json.dumps({"cookies": [{"name": "sid"}], "origins": []}))
        return str(p)

    monkeypatch.setattr(glints_auth, "_login_and_save", _fake_login)
    assert await glints_auth.ensure_session() == str(p)
    assert p.is_file()


@pytest.mark.asyncio
async def test_ensure_session_anonymous_when_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "glints_storage_state_path", str(tmp_path / "nope.json"))
    monkeypatch.setattr(settings, "glints_email", "")
    monkeypatch.setattr(settings, "glints_password", "")
    assert await glints_auth.ensure_session() is None


@pytest.mark.asyncio
async def test_ensure_session_returns_none_when_login_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "glints_storage_state_path", str(tmp_path / "x.json"))
    monkeypatch.setattr(settings, "glints_email", "bot@example.com")
    monkeypatch.setattr(settings, "glints_password", "pw")

    async def _fail_login():
        return None

    monkeypatch.setattr(glints_auth, "_login_and_save", _fail_login)
    assert await glints_auth.ensure_session() is None


# --------------------------------------------------------------------------- #
# base.apply_stealth — the silent-skip of this helper was the root cause of the
# headless-login Cloudflare timeout, so its contract is pinned by tests.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_apply_stealth_uses_v2_class_when_available(monkeypatch):
    """playwright-stealth v2 exposes a `Stealth` class; apply_stealth must use
    its per-page `apply_stealth_async` (init_scripts_only) and report success."""
    import playwright_stealth

    from app.scrapers import base

    called: dict = {}

    class FakeStealth:
        def __init__(self, **kwargs):
            called["init"] = kwargs

        async def apply_stealth_async(self, page):
            called["page"] = page

    monkeypatch.setattr(playwright_stealth, "Stealth", FakeStealth)
    sentinel = object()
    assert await base.apply_stealth(sentinel) is True
    assert called["page"] is sentinel
    assert called["init"].get("init_scripts_only") is True


@pytest.mark.asyncio
async def test_apply_stealth_degrades_gracefully_on_error(monkeypatch):
    """A broken stealth backend must never propagate — it returns False so the
    browser keeps working (the previous code swallowed even the ImportError,
    silently running with no stealth at all)."""
    import playwright_stealth

    from app.scrapers import base

    class BoomStealth:
        def __init__(self, **kwargs): ...

        async def apply_stealth_async(self, page):
            raise RuntimeError("stealth backend exploded")

    monkeypatch.setattr(playwright_stealth, "Stealth", BoomStealth)
    assert await base.apply_stealth(object()) is False


# --------------------------------------------------------------------------- #
# base.browser_session storage_state passthrough
# --------------------------------------------------------------------------- #
class _FakePage:
    async def goto(self, *a, **k): ...
    async def content(self):
        return ""


class _FakeContext:
    def __init__(self, rec):
        self._rec = rec

    async def new_page(self):
        return _FakePage()

    async def close(self): ...


class _FakeBrowser:
    def __init__(self, rec):
        self._rec = rec

    async def new_context(self, **kwargs):
        self._rec["context_kwargs"] = kwargs
        return _FakeContext(self._rec)

    async def close(self): ...


class _FakeChromium:
    def __init__(self, rec):
        self._rec = rec

    async def launch(self, **kwargs):
        return _FakeBrowser(self._rec)


class _FakePW:
    def __init__(self, rec):
        self.chromium = _FakeChromium(rec)


class _FakePWCM:
    def __init__(self, rec):
        self._rec = rec

    async def __aenter__(self):
        return _FakePW(self._rec)

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_browser_session_passes_storage_state(monkeypatch):
    rec: dict = {}
    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: _FakePWCM(rec))
    from app.scrapers.base import browser_session

    async with browser_session("glints", storage_state="/tmp/glints_state.json"):
        pass
    assert rec["context_kwargs"].get("storage_state") == "/tmp/glints_state.json"


@pytest.mark.asyncio
async def test_browser_session_omits_storage_state_when_none(monkeypatch):
    rec: dict = {}
    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: _FakePWCM(rec))
    from app.scrapers.base import browser_session

    async with browser_session("glints"):
        pass
    assert "storage_state" not in rec["context_kwargs"]


# --------------------------------------------------------------------------- #
# GlintsScraper auth integration
# --------------------------------------------------------------------------- #
def _gql_body(page: int) -> str:
    return json.dumps(
        {
            "operationName": "searchJobsV3",
            "variables": {
                "data": {"SearchTerm": "x", "CountryCode": "ID", "pageSize": 30, "page": page}
            },
            "query": "query searchJobsV3 { searchJobsV3 { jobsInPage { id } } }",
        }
    )


def _jobs_payload(page_num: int, n: int = 10) -> dict:
    return {
        "data": {
            "searchJobsV3": {
                "jobsInPage": [
                    {"id": f"p{page_num}-{i}", "title": f"Job {i}", "company": {"name": "Co"}}
                    for i in range(n)
                ]
            }
        }
    }


@pytest.mark.asyncio
async def test_scrape_forwards_authorization_header_on_replay(monkeypatch):
    """The bearer token from the authenticated page-1 request must be forwarded
    on the replayed pagination POSTs; the raw cookie header must not be."""
    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)
    s = GlintsScraper()
    sent_headers: list[dict] = []

    async def on_event(_ev): ...

    class FakeReq:
        method = "POST"
        post_data = _gql_body(1)

        async def all_headers(self):
            return {
                "authorization": "Bearer tok123",
                "x-glints-country-code": "ID",
                "accept-language": "id",
                "cookie": "sid=secret",
            }

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

        def on(self, _e, cb):
            self._handlers.append(cb)

        async def goto(self, *a, **k):
            await self._handlers[0](
                FakeResp(
                    "https://glints.com/api/v2-alc/graphql?op=searchJobsV3",
                    _jobs_payload(1),
                )
            )

        async def content(self):
            return ""

        async def evaluate(self, _script, arg):
            _url, _body, headers = arg
            sent_headers.append(headers)
            return (
                _jobs_payload(2)
                if json.loads(_body)["variables"]["data"]["page"] == 2
                else {"data": {"searchJobsV3": {"jobsInPage": []}}}
            )

    params = SearchParams(role_keywords=["x"], location=["Jakarta"], work_type=["remote"])
    await s.scrape(FakePage(), params, on_event)

    assert sent_headers, "expected at least one replayed POST"
    h = sent_headers[0]
    assert h.get("authorization") == "Bearer tok123"
    assert h.get("x-glints-country-code") == "ID"
    assert "cookie" not in h  # cookies travel via credentials:'include', not headers


@pytest.mark.asyncio
async def test_scrape_reauths_once_when_session_expired(monkeypatch):
    """A NO_PERMISSION page-2 reply should trigger exactly one session refresh,
    apply the new cookies to the context, and retry the page to recover jobs."""
    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    refresh_calls = {"n": 0}

    async def fake_refresh():
        refresh_calls["n"] += 1
        return "/tmp/glints_state.json"

    monkeypatch.setattr("app.scrapers.glints_auth.refresh_session", fake_refresh)
    monkeypatch.setattr(
        "app.scrapers.glints._session_cookies_and_token",
        lambda _p: (
            [{"name": "sid", "value": "fresh", "domain": ".glints.com", "path": "/"}],
            None,
        ),
    )

    s = GlintsScraper()
    added_cookies: list = []

    async def on_event(_ev): ...

    class FakeReq:
        method = "POST"
        post_data = _gql_body(1)

        async def all_headers(self):
            return {"x-glints-country-code": "ID"}

    class FakeResp:
        def __init__(self, url, body):
            self.url = url
            self.headers = {"content-type": "application/json"}
            self._body = body
            self.request = FakeReq()

        async def json(self):
            return self._body

    class FakeContext:
        async def add_cookies(self, cookies):
            added_cookies.extend(cookies)

    class FakePage:
        def __init__(self):
            self._handlers: list = []
            self._p2_seen = 0
            self.context = FakeContext()

        def on(self, _e, cb):
            self._handlers.append(cb)

        async def goto(self, *a, **k):
            await self._handlers[0](
                FakeResp(
                    "https://glints.com/api/v2-alc/graphql?op=searchJobsV3",
                    _jobs_payload(1),
                )
            )

        async def content(self):
            return ""

        async def evaluate(self, _script, arg):
            page_num = json.loads(arg[1])["variables"]["data"]["page"]
            if page_num == 2:
                self._p2_seen += 1
                if self._p2_seen == 1:
                    return {"errors": [{"extensions": {"code": "NO_PERMISSION"}}]}
                return _jobs_payload(2)
            return {"data": {"searchJobsV3": {"jobsInPage": []}}}

    params = SearchParams(role_keywords=["x"], location=["Jakarta"], work_type=["remote"])
    jobs = await s.scrape(FakePage(), params, on_event)

    assert refresh_calls["n"] == 1
    assert added_cookies and added_cookies[0]["value"] == "fresh"
    assert len(jobs) == 20  # 10 from page 1 + 10 from the retried page 2


@pytest.mark.asyncio
async def test_scrape_falls_back_to_page1_when_reauth_unavailable(monkeypatch):
    """If the session is gated and refresh yields nothing, return page-1 jobs
    and emit a warning event instead of crashing."""
    monkeypatch.setattr("app.scrapers.glints.humanizer_delay", lambda _: 0)

    async def fake_refresh():
        return None

    monkeypatch.setattr("app.scrapers.glints_auth.refresh_session", fake_refresh)

    s = GlintsScraper()
    events: list[dict] = []

    async def on_event(ev):
        events.append(ev)

    class FakeReq:
        method = "POST"
        post_data = _gql_body(1)

        async def all_headers(self):
            return {"x-glints-country-code": "ID"}

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

        def on(self, _e, cb):
            self._handlers.append(cb)

        async def goto(self, *a, **k):
            await self._handlers[0](
                FakeResp(
                    "https://glints.com/api/v2-alc/graphql?op=searchJobsV3",
                    _jobs_payload(1),
                )
            )

        async def content(self):
            return ""

        async def evaluate(self, _script, _arg):
            return {"errors": [{"extensions": {"code": "NO_PERMISSION"}}]}

    params = SearchParams(role_keywords=["x"], location=["Jakarta"], work_type=["remote"])
    jobs = await s.scrape(FakePage(), params, on_event)

    assert len(jobs) == 10  # page 1 retained
    assert any(
        e.get("severity") == "warning" and "session expired" in e.get("message", "").lower()
        for e in events
    )
