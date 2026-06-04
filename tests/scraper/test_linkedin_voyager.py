import json

from app.scrapers import linkedin_voyager as lv


def _state_file(tmp_path, jsessionid_value):
    p = tmp_path / "linkedin_state.json"
    p.write_text(
        json.dumps(
            {
                "cookies": [
                    {"name": "li_at", "value": "AQED-token", "domain": ".linkedin.com"},
                    {
                        "name": "JSESSIONID",
                        "value": jsessionid_value,
                        "domain": ".www.linkedin.com",
                    },
                ],
                "origins": [],
            }
        )
    )
    return str(p)


def test_csrf_token_from_state_strips_surrounding_quotes(tmp_path):
    # LinkedIn stores JSESSIONID as a quoted value; csrf-token is the unquoted form.
    path = _state_file(tmp_path, '"ajax:1234567890"')
    assert lv.csrf_token_from_state(path) == "ajax:1234567890"


def test_csrf_token_from_state_returns_none_without_jsessionid(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"cookies": [{"name": "li_at", "value": "x"}], "origins": []}))
    assert lv.csrf_token_from_state(str(p)) is None


def test_voyager_headers_includes_csrf_restli_and_normalized_accept(tmp_path):
    path = _state_file(tmp_path, '"ajax:abc"')
    h = lv.voyager_headers(path)
    assert h["csrf-token"] == "ajax:abc"
    assert h["x-restli-protocol-version"] == "2.0.0"
    # Voyager rejects application/json with HTTP 400; it requires its normalized
    # content type (confirmed from a real browser request).
    assert h["accept"] == "application/vnd.linkedin.normalized+json+2.1"
    assert "clientVersion" in h["x-li-track"]
    assert h["x-li-page-instance"].startswith("urn:li:page:")


def test_voyager_headers_empty_when_no_token(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"cookies": [], "origins": []}))
    assert lv.voyager_headers(str(p)) == {}


def test_voyager_detail_url_contains_job_id():
    url = lv.voyager_detail_url("4413287663")
    assert url.startswith("https://www.linkedin.com/voyager/api/")
    assert "4413287663" in url


def test_voyager_search_url_uses_graphql_jobcards_query():
    url = lv.voyager_search_url("python backend", "Jakarta", start=0)
    # Job search is the Voyager GraphQL endpoint with a versioned queryId and a
    # RestLi tuple `variables=(...)` (confirmed from a real browser request).
    assert url.startswith("https://www.linkedin.com/voyager/api/graphql?")
    assert "queryId=voyagerJobsDashJobCards." in url
    assert "variables=(count:25,query:(origin:JOBS_HOME_SEARCH_BUTTON,keywords:" in url
    assert "python%20backend" in url  # keywords percent-encoded
    assert "Jakarta" in url  # location folded into the keywords term
    assert "start:0)" in url
    assert " " not in url  # no raw spaces anywhere in the URL


def test_voyager_search_url_encodes_slash_in_keywords():
    # A "/" in a keyword must be percent-encoded in the query, not left as a
    # path separator the Voyager parser could misread.
    url = lv.voyager_search_url("java/kotlin", "Jakarta")
    assert "java%2Fkotlin" in url


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body


class _FakeRequest:
    def __init__(self):
        self.calls = []  # (url, headers)

    async def get(self, url, headers=None):
        self.calls.append((url, headers or {}))
        if "voyagerJobsDashJobCards" in url:
            return _FakeResp(200, '{"search":"ok"}')
        return _FakeResp(200, '{"detail":"ok"}')


class _FakePage:
    def __init__(self):
        self.request = _FakeRequest()


class _FakeDump:
    def __init__(self):
        self.listings = []
        self.details = []

    def add_listing(self, page, url, data, post_data=None):
        self.listings.append((page, url, data))

    def add_detail(self, job_id, data):
        self.details.append((job_id, data))


async def test_capture_voyager_dumps_search_and_detail_with_headers(tmp_path):
    path = _state_file(tmp_path, '"ajax:tok"')
    page = _FakePage()
    dump = _FakeDump()

    await lv.capture_voyager(
        page,
        storage_state_path=path,
        keywords="python",
        location="Jakarta",
        job_ids=["111", "222"],
        dump=dump,
        max_details=2,
    )

    # search dumped once, details dumped per id (capped by max_details)
    assert len(dump.listings) == 1
    assert {d[0] for d in dump.details} == {"111", "222"}
    # every Voyager request carried the csrf-token header
    assert all(h.get("csrf-token") == "ajax:tok" for _u, h in page.request.calls)
    import json as _json

    assert _json.loads(dump.listings[0][2])["status"] == 200


async def test_capture_voyager_noop_without_token(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"cookies": [], "origins": []}')
    page = _FakePage()
    dump = _FakeDump()
    await lv.capture_voyager(
        page,
        storage_state_path=str(p),
        keywords="x",
        location="y",
        job_ids=["1"],
        dump=dump,
    )
    assert page.request.calls == []  # no token → no Voyager calls


async def test_capture_voyager_never_raises_when_dump_raises(tmp_path):
    path = _state_file(tmp_path, '"ajax:tok"')
    page = _FakePage()

    class _BoomDump:
        def add_listing(self, *a, **k):
            raise RuntimeError("dump exploded")

        def add_detail(self, *a, **k):
            raise RuntimeError("dump exploded")

    # Must NOT raise — capture is best-effort instrumentation.
    await lv.capture_voyager(
        page,
        storage_state_path=path,
        keywords="x",
        location="y",
        job_ids=["1"],
        dump=_BoomDump(),
    )
