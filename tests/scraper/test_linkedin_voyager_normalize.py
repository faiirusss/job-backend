import json
import pathlib

from app.scrapers import linkedin_voyager_normalize as vn

FIX = pathlib.Path(__file__).parent.parent / "fixtures"
SEARCH = (FIX / "linkedin_voyager_search.json").read_text(encoding="utf-8")
DETAIL = (FIX / "linkedin_voyager_detail.json").read_text(encoding="utf-8")
# Rich real-world detail: nested bullet lists, bold/italic, and an originalListedAt
# (2025-06-10) distinct from its repost listedAt (2026-05-17).
DETAIL_RICH = (FIX / "linkedin_voyager_detail_rich.json").read_text(encoding="utf-8")


def _detail_doc(**data_overrides):
    """A minimal Voyager detail body with a one-line description, plus overrides
    folded into ``data`` — for asserting individual field-mapping rules."""
    return json.dumps({"data": {"description": {"text": "Hello"}, **data_overrides}})


def _attr(start, length, kind):
    """Build a single Pemberly attribute. ``kind`` is the unqualified type name
    (e.g. "Bold", "ListItem"); ``List`` accepts an ``ordered`` flag via a tuple."""
    ordered = None
    if isinstance(kind, tuple):
        kind, ordered = kind
    full = f"com.linkedin.pemberly.text.{kind}"
    union: dict = {"$type": full}
    if ordered is not None:
        union["ordered"] = ordered
    key = kind[0].lower() + kind[1:]
    return {
        "start": start,
        "length": length,
        "type": {"$type": full},
        "attributeKindUnion": {key: union},
    }


def test_parse_voyager_search_extracts_cards():
    jobs = vn.parse_voyager_search(SEARCH)
    assert len(jobs) == 3
    j = next(x for x in jobs if x.id == "4277612327")
    assert j.portal == "linkedin"
    assert j.title == "Java Full Stack Developer"
    assert j.company == "PT IKONSULTAN INOVATAMA"
    assert j.location == "Gambir, Jakarta, Indonesia"  # work-type paren stripped
    assert j.work_type == "onsite"
    assert j.apply_url == "https://www.linkedin.com/jobs/view/4277612327"
    assert j.posted_date == "2025-07-29"  # 1753780902000 ms → UTC date
    assert j.company_logo_url and j.company_logo_url.startswith("https://media.licdn.com/")


def test_parse_voyager_search_empty_returns_empty():
    assert vn.parse_voyager_search('{"included": []}') == []
    assert vn.parse_voyager_search("") == []


def test_parse_voyager_detail_extracts_rich_fields():
    parsed = vn.parse_voyager_detail(DETAIL, "4400626671")
    assert parsed["description"].startswith("Experience IT Developer")
    assert parsed["seniority"] == "senior"  # "Mid-Senior level"
    nj = parsed["detail"]
    assert nj.job_type == "FULL_TIME"  # "Full-time" — via detail, not top-level key
    assert nj.title == "IT Developer"
    assert nj.job_type_label  # non-empty human label
    assert "Engineering" in nj.category.breadcrumb
    assert "IT Services and IT Consulting" in nj.category.breadcrumb
    assert nj.description_html.startswith("<p>")
    assert nj.location.name == "Jakarta Metropolitan Area"


def test_parse_voyager_detail_empty_without_description():
    assert vn.parse_voyager_detail('{"data": {}}', "1") == {}
    assert vn.parse_voyager_detail("", "1") == {}


# --- #1 relative posting time: posted_at from originalListedAt -----------------


def test_posted_at_prefers_original_listed_date():
    # Rich fixture: originalListedAt 1749579832000 → 2025-06-10 (the "1 year ago"
    # LinkedIn shows), NOT its 2026-05-17 repost listedAt.
    nj = vn.parse_voyager_detail(DETAIL_RICH, "4246297851")["detail"]
    assert nj.posted_at is not None
    assert nj.posted_at.startswith("2025-06-10")


def test_posted_at_falls_back_listed_then_created_then_none():
    listed = vn.parse_voyager_detail(_detail_doc(listedAt=1749579832000), "1")["detail"]
    assert listed.posted_at.startswith("2025-06-10")
    created = vn.parse_voyager_detail(_detail_doc(createdAt=1749579832000), "1")["detail"]
    assert created.posted_at.startswith("2025-06-10")
    assert vn.parse_voyager_detail(_detail_doc(), "1")["detail"].posted_at is None


# --- #5 salary: only surfaced when present ------------------------------------


def test_salary_shown_only_when_present():
    sal = vn.parse_voyager_detail(
        _detail_doc(formattedSalaryDescription="IDR10M/month - IDR15M/month"), "1"
    )["detail"].salary
    assert sal.show is True
    assert sal.label == "IDR10M/month - IDR15M/month"

    empty = vn.parse_voyager_detail(_detail_doc(formattedSalaryDescription=""), "1")[
        "detail"
    ].salary
    assert empty.show is False
    assert vn.parse_voyager_detail(_detail_doc(), "1")["detail"].salary.show is False


# --- #4 description formatting: Pemberly attributed text → HTML ----------------


def test_render_attributed_unordered_list():
    desc = {
        "text": "Intro\n\nFirst\nSecond",
        "attributes": [
            _attr(7, 12, ("List", False)),
            _attr(7, 5, "ListItem"),
            _attr(13, 6, "ListItem"),
        ],
    }
    out = vn._render_attributed(desc)
    assert "<ul>" in out and "</ul>" in out
    assert out.count("<li>") == 2
    assert "<li>First</li>" in out
    assert "<p>Intro</p>" in out


def test_render_attributed_ordered_list():
    desc = {
        "text": "A\nB",
        "attributes": [
            _attr(0, 3, ("List", True)),
            _attr(0, 1, "ListItem"),
            _attr(2, 1, "ListItem"),
        ],
    }
    out = vn._render_attributed(desc)
    assert "<ol>" in out and "</ol>" in out


def test_render_attributed_inline_bold_italic():
    desc = {"text": "a bold word", "attributes": [_attr(2, 4, "Bold"), _attr(7, 4, "Italic")]}
    out = vn._render_attributed(desc)
    assert "<strong>bold</strong>" in out
    assert "<em>word</em>" in out


def test_render_attributed_escapes_and_breaks():
    desc = {"text": "a <b> & c\nnext line", "attributes": [_attr(9, 1, "LineBreak")]}
    out = vn._render_attributed(desc)
    assert "&lt;b&gt;" in out and "&amp;" in out
    assert "<br>" in out and "<b>" not in out


def test_render_attributed_no_attributes_falls_back_to_paragraphs():
    desc = {"text": "Para one\n\nPara two", "attributes": []}
    out = vn._render_attributed(desc)
    assert out == "<p>Para one</p><p>Para two</p>"


def test_rich_fixture_description_html_has_bulleted_list():
    nj = vn.parse_voyager_detail(DETAIL_RICH, "4246297851")["detail"]
    assert "<ul>" in nj.description_html
    assert "<li>" in nj.description_html
    assert "<strong>" in nj.description_html
