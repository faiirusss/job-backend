import json

import pytest

from app.scrapers import linkedin_login as seed
from app.scrapers.linkedin_voyager import csrf_token_from_state


def test_build_storage_state_has_both_cookies_on_right_domains():
    state = seed.build_storage_state("AQED-tok", '"ajax:1234"')
    by_name = {c["name"]: c for c in state["cookies"]}
    assert by_name["li_at"]["value"] == "AQED-tok"
    assert by_name["li_at"]["domain"] == ".linkedin.com"
    assert by_name["JSESSIONID"]["domain"] == ".www.linkedin.com"
    assert state["origins"] == []


def test_build_storage_state_normalizes_jsessionid_quotes():
    # Whether the user pastes it quoted or bare, the stored cookie is quoted
    # (LinkedIn's on-the-wire form) so the server accepts it.
    quoted = seed.build_storage_state("x", '"ajax:99"')
    bare = seed.build_storage_state("x", "ajax:99")
    jq = next(c for c in quoted["cookies"] if c["name"] == "JSESSIONID")
    jb = next(c for c in bare["cookies"] if c["name"] == "JSESSIONID")
    assert jq["value"] == '"ajax:99"'
    assert jb["value"] == '"ajax:99"'


def test_written_state_yields_unquoted_csrf_token(tmp_path):
    # End-to-end with the Voyager helper: the written file must produce the
    # csrf-token (unquoted) that voyager_headers needs.
    out = tmp_path / "linkedin_state.json"
    seed.write_storage_state("AQED-tok", "ajax:abc", str(out))
    assert json.loads(out.read_text())  # valid JSON
    assert csrf_token_from_state(str(out)) == "ajax:abc"


@pytest.mark.parametrize("li_at,jsid", [("", "ajax:1"), ("tok", ""), ("  ", "ajax:1")])
def test_build_storage_state_requires_both(li_at, jsid):
    with pytest.raises(ValueError):
        seed.build_storage_state(li_at, jsid)
