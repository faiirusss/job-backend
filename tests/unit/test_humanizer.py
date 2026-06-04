import pytest

from app.scrapers.base import humanizer_delay, random_user_agent


def test_humanizer_delay_within_bounds():
    for _ in range(50):
        d = humanizer_delay("page_load")
        assert 2.0 <= d <= 5.0


def test_humanizer_delay_click():
    for _ in range(50):
        d = humanizer_delay("click")
        assert 1.0 <= d <= 3.0


def test_humanizer_delay_unknown_raises():
    with pytest.raises(KeyError):
        humanizer_delay("nope")


def test_random_user_agent_is_chrome_ish():
    ua = random_user_agent()
    assert "Chrome" in ua or "Mozilla" in ua
