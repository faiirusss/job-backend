import json

import pytest

from app.config import settings
from app.scrapers import linkedin_auth


def test_storage_state_path_returns_path_for_valid_file(tmp_path, monkeypatch):
    p = tmp_path / "linkedin_state.json"
    p.write_text(json.dumps({"cookies": [], "origins": []}))
    monkeypatch.setattr(settings, "linkedin_storage_state_path", str(p))
    assert linkedin_auth.storage_state_path() == str(p)


def test_storage_state_path_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "linkedin_storage_state_path", str(tmp_path / "nope.json"))
    assert linkedin_auth.storage_state_path() is None


def test_storage_state_path_returns_none_for_invalid_json(tmp_path, monkeypatch):
    p = tmp_path / "linkedin_state.json"
    p.write_text("{not json")
    monkeypatch.setattr(settings, "linkedin_storage_state_path", str(p))
    assert linkedin_auth.storage_state_path() is None


def test_has_credentials(monkeypatch):
    monkeypatch.setattr(settings, "linkedin_email", "a@b.com")
    monkeypatch.setattr(settings, "linkedin_password", "pw")
    assert linkedin_auth.has_credentials() is True
    monkeypatch.setattr(settings, "linkedin_password", "")
    assert linkedin_auth.has_credentials() is False


async def test_refresh_session_noop_without_credentials(monkeypatch):
    monkeypatch.setattr(settings, "linkedin_email", "")
    monkeypatch.setattr(settings, "linkedin_password", "")
    assert await linkedin_auth.refresh_session() is None


async def test_ensure_session_returns_existing_without_login(tmp_path, monkeypatch):
    p = tmp_path / "linkedin_state.json"
    p.write_text(json.dumps({"cookies": [], "origins": []}))
    monkeypatch.setattr(settings, "linkedin_storage_state_path", str(p))

    called = {"login": False}

    async def _no_login():
        called["login"] = True
        return None

    monkeypatch.setattr(linkedin_auth, "_login_and_save", _no_login)
    assert await linkedin_auth.ensure_session() == str(p)
    assert called["login"] is False


async def test_ensure_session_auto_seeds_when_missing(tmp_path, monkeypatch):
    p = tmp_path / "linkedin_state.json"
    monkeypatch.setattr(settings, "linkedin_storage_state_path", str(p))
    monkeypatch.setattr(settings, "linkedin_email", "bot@example.com")
    monkeypatch.setattr(settings, "linkedin_password", "pw")

    async def _fake_login():
        p.write_text(json.dumps({"cookies": [{"name": "li_at"}], "origins": []}))
        return str(p)

    monkeypatch.setattr(linkedin_auth, "_login_and_save", _fake_login)
    assert await linkedin_auth.ensure_session() == str(p)
    assert p.is_file()


async def test_ensure_session_anonymous_when_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "linkedin_storage_state_path", str(tmp_path / "nope.json"))
    monkeypatch.setattr(settings, "linkedin_email", "")
    monkeypatch.setattr(settings, "linkedin_password", "")
    assert await linkedin_auth.ensure_session() is None


async def test_ensure_session_returns_none_when_login_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "linkedin_storage_state_path", str(tmp_path / "x.json"))
    monkeypatch.setattr(settings, "linkedin_email", "bot@example.com")
    monkeypatch.setattr(settings, "linkedin_password", "pw")

    async def _fail_login():
        return None

    monkeypatch.setattr(linkedin_auth, "_login_and_save", _fail_login)
    assert await linkedin_auth.ensure_session() is None


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.linkedin.com/feed/", False),
        ("https://www.linkedin.com/checkpoint/challenge/", True),
        ("https://www.linkedin.com/uas/login-submit", True),
        ("https://www.linkedin.com/challenge/", True),
        ("https://www.linkedin.com/feed/challenge-hiring/", False),
    ],
)
def test_is_checkpoint_url(url, expected):
    assert linkedin_auth._is_checkpoint_url(url) is expected


@pytest.mark.parametrize(
    "title,body,expected_substr",
    [
        ("Security Verification", "Please solve this CAPTCHA to continue", "anti-bot"),
        ("LinkedIn", "We value your privacy. Accept cookies to continue.", "cookie-consent"),
        ("Security Check", "Enter the code we sent for two-step verification", "checkpoint"),
        ("LinkedIn: Log In or Sign Up", "New to LinkedIn? Join now to make the most", "signup"),
        ("Something else", "totally unrelated content", "unrecognized"),
    ],
)
def test_diagnose_login_page_classifies(title, body, expected_substr):
    diag = linkedin_auth._diagnose_login_page("https://www.linkedin.com/login", title, body)
    assert expected_substr in diag.lower()


class _FakeEl:
    def __init__(self, visible: bool):
        self._visible = visible

    async def is_visible(self) -> bool:
        return self._visible


class _FakeLocator:
    def __init__(self, els):
        self._els = els

    async def count(self) -> int:
        return len(self._els)

    def nth(self, i: int):
        return self._els[i]


class _FakeLocPage:
    def __init__(self, els):
        self._els = els

    def locator(self, _selector):
        return _FakeLocator(self._els)


async def test_first_visible_skips_hidden_duplicate():
    # LinkedIn's new login renders duplicate input variants; the first match can
    # be the hidden one, so _first_visible must return the first VISIBLE element.
    hidden, visible = _FakeEl(False), _FakeEl(True)
    el = await linkedin_auth._first_visible(_FakeLocPage([hidden, visible]), "input")
    assert el is visible


async def test_first_visible_returns_none_when_all_hidden():
    page = _FakeLocPage([_FakeEl(False), _FakeEl(False)])
    assert await linkedin_auth._first_visible(page, "input") is None


@pytest.mark.parametrize(
    "name,should_match",
    [
        ("Login", True),
        ("Masuk", True),
        ("Sign in", True),
        ("LOGIN", True),
        ("masuk", True),
        # The primary submit must NOT be confused with the social-login buttons.
        ("Masuk dengan Microsoft", False),
        ("Continue with Google", False),
        ("Sign in with Google", False),
        ("Lupa kata sandi?", False),
    ],
)
def test_submit_name_re_matches_only_primary_button(name, should_match):
    assert bool(linkedin_auth._SUBMIT_NAME_RE.search(name)) is should_match
