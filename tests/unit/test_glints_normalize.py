import json
import pathlib
import re

from app.scrapers import glints_normalize as gn


def _draft(*blocks: dict) -> str:
    return json.dumps({"blocks": list(blocks), "entityMap": {}})


def _b(text: str, type_: str = "unstyled", styles=None) -> dict:
    return {
        "text": text,
        "type": type_,
        "depth": 0,
        "entityRanges": [],
        "inlineStyleRanges": styles or [],
        "key": "k",
    }


def test_blocks_to_html_paragraph_and_escaping():
    html = gn.parse_draftjs(_draft(_b('a < b & "c"')))
    assert html == "<p>a &lt; b &amp; &quot;c&quot;</p>"


def test_blocks_to_html_groups_consecutive_list_items():
    html = gn.parse_draftjs(
        _draft(
            _b("one", "ordered-list-item"),
            _b("two", "ordered-list-item"),
            _b("para"),
            _b("a", "unordered-list-item"),
        )
    )
    assert html == "<ol><li>one</li><li>two</li></ol><p>para</p><ul><li>a</li></ul>"


def test_inline_styles_bold_italic():
    html = gn.parse_draftjs(
        _draft(
            _b(
                "hello",
                styles=[{"offset": 0, "length": 5, "style": "BOLD"}],
            )
        )
    )
    assert html == "<p><strong>hello</strong></p>"


def test_headers_and_blank_blocks_skipped():
    html = gn.parse_draftjs(_draft(_b("Title", "header-two"), _b("   ")))
    assert html == "<h2>Title</h2>"


def test_parse_draftjs_split_at_requirements_heading():
    desc, req = gn.parse_draftjs_split(
        _draft(
            _b("Job Descriptions"),
            _b("Do things"),
            _b("Job Requirements"),
            _b("Know things"),
        )
    )
    assert "Do things" in desc and "Job Requirements" not in desc
    assert req is not None and "Know things" in req


def test_parse_draftjs_split_no_heading_returns_none():
    desc, req = gn.parse_draftjs_split(_draft(_b("Only desc")))
    assert "Only desc" in desc
    assert req is None


def test_draftjs_to_text_joins_nonempty_blocks():
    text = gn.draftjs_to_text(_draft(_b("a"), _b(" "), _b("b")))
    assert text == "a\nb"


def test_draftjs_helpers_tolerate_garbage():
    assert gn.parse_draftjs(None) == ""
    assert gn.parse_draftjs("not json") == ""
    assert gn.parse_draftjs_split(None) == ("", None)
    assert gn.draftjs_to_text("{bad") == ""


def test_enum_label_known_and_unknown():
    assert gn.enum_label("FULL_TIME", gn.JOB_TYPE_LABEL) == "Penuh Waktu"
    assert gn.enum_label("ONSITE", gn.WORK_ARRANGEMENT_LABEL) == "Kerja di lokasi"
    assert gn.enum_label("DIPLOMA", gn.EDUCATION_LABEL) == "Diploma (D1–D4)"
    # Unknown enum -> Title Case fallback, never crashes
    assert gn.enum_label("SOME_NEW_TYPE", gn.JOB_TYPE_LABEL) == "Some New Type"
    assert gn.enum_label(None, gn.JOB_TYPE_LABEL) == ""


def test_experience_label_boundaries():
    assert "fresh graduate" in gn.experience_label(0, 1)
    assert gn.experience_label(2, 4) == "2–4 tahun pengalaman"
    assert "Lebih dari 10" in gn.experience_label(10, 15)


def test_build_salary_shown_range():
    s = gn.build_salary(
        {
            "shouldShowSalary": True,
            "salaries": [
                {
                    "CurrencyCode": "IDR",
                    "minAmount": 8000000,
                    "maxAmount": 10000000,
                    "salaryMode": "MONTH",
                }
            ],
        }
    )
    assert s["show"] is True
    assert s["min"] == 8000000 and s["max"] == 10000000
    assert s["label"] == "IDR 8.000.000 – IDR 10.000.000/bulan"


def test_build_salary_hidden():
    s = gn.build_salary({"shouldShowSalary": False, "salaries": None})
    assert s["show"] is False
    assert s["label"] == "Gaji tidak ditampilkan"


def test_parse_social_media_skips_null():
    out = gn.parse_social_media('{"linkedin":"95753740","instagram":null}')
    assert out == [{"platform": "linkedin", "url": "https://www.linkedin.com/company/95753740"}]
    assert gn.parse_social_media(None) == []
    assert gn.parse_social_media("garbage") == []


def test_parse_gallery_builds_urls():
    out = gn.parse_gallery('["a.jpg","b.jpg"]')
    assert out == [
        "https://glints-dashboard.oss-ap-southeast-1.aliyuncs.com/company-photo/a.jpg",
        "https://glints-dashboard.oss-ap-southeast-1.aliyuncs.com/company-photo/b.jpg",
    ]
    assert gn.parse_gallery(None) == []


def test_build_image_url_passthrough_and_none():
    assert gn.build_image_url(None) is None
    assert gn.build_image_url("https://x/y.png") == "https://x/y.png"
    assert gn.build_image_url("x.png", kind="company-logo").endswith("/company-logo/x.png")


def test_build_urls_prefers_external_apply():
    canon, apply = gn.build_urls(
        {"id": "abc", "title": "Fullstack Engineer", "externalApplyURL": "https://ext/apply"}
    )
    assert canon == "https://glints.com/id/opportunities/jobs/fullstack-engineer/abc"
    assert apply == "https://ext/apply"
    canon2, apply2 = gn.build_urls({"id": "abc", "title": "X"})
    assert apply2 == canon2


def test_blocks_tolerate_non_dict_entries():
    import json as _json

    raw = _json.dumps(
        {
            "blocks": [
                "a string",
                None,
                {
                    "text": "ok",
                    "type": "unstyled",
                    "depth": 0,
                    "entityRanges": [],
                    "inlineStyleRanges": [],
                    "key": "k",
                },
            ]
        }
    )
    assert gn.parse_draftjs(raw) == "<p>ok</p>"


def test_negative_offset_inline_style_ignored():
    raw = _draft(_b("hi", styles=[{"offset": -1, "length": 2, "style": "BOLD"}]))
    assert gn.parse_draftjs(raw) == "<p>hi</p>"


def test_blank_list_item_skipped():
    raw = _draft(_b("one", "ordered-list-item"), _b("  ", "ordered-list-item"))
    assert gn.parse_draftjs(raw) == "<ol><li>one</li></ol>"


def test_build_salary_shown_but_no_amounts_is_hidden():
    s = gn.build_salary(
        {
            "shouldShowSalary": True,
            "salaries": [{"CurrencyCode": "IDR", "minAmount": None, "maxAmount": None}],
        }
    )
    assert s["show"] is False
    assert s["label"] == "Gaji tidak ditampilkan"


def test_normalized_job_schema_defaults():
    from app.schemas import NormalizedJob

    nj = NormalizedJob(id="x", title="T")
    assert nj.skills == []
    assert nj.salary.show is False
    assert nj.company.name == ""
    # JSON round-trips (mode="json") for JSONB persistence
    dumped = nj.model_dump(mode="json")
    again = NormalizedJob.model_validate(dumped)
    assert again.id == "x"


_FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "glints_next_data.json"


def _next_data() -> dict:
    return json.loads(_FIXTURE.read_text())


def test_extract_glints_job_core_fields():
    nj = gn.extract_glints_job(_next_data())
    assert nj is not None
    assert nj.title == "Fullstack Engineer"
    assert nj.company.name == "Nodewave Indonesia"
    assert nj.company.is_verified is True
    assert nj.job_type_label == "Penuh Waktu"
    assert nj.work_arrangement_label == "Kerja di lokasi"
    assert nj.education_level_label == "Diploma (D1–D4)"
    # Skills carry mustHave
    assert any(s.must_have for s in nj.skills)
    assert any(s.name == "Node.js" for s in nj.skills)
    # Location
    assert nj.location.city == "Jakarta Barat"
    assert nj.location.province == "DKI Jakarta"
    assert nj.location.country == "Indonesia"
    # Benefits structured
    assert len(nj.benefits) == 7
    assert nj.benefits[0].title
    # Description split rendered as HTML
    assert nj.description_html.startswith("<")
    # Category breadcrumb ordered, main name last
    assert nj.category.name == "Full Stack Developer"
    assert nj.category.breadcrumb[-1] == "Full Stack Developer"
    # Company gallery URLs built
    assert all(u.startswith("https://") for u in nj.company.gallery_urls)


def test_extract_glints_job_missing_root_returns_none():
    assert gn.extract_glints_job({}) is None
    assert gn.extract_glints_job({"props": {"pageProps": {}}}) is None


def test_extract_glints_job_leaks_no_secrets():
    blob = gn.extract_glints_job(_next_data()).model_dump_json()
    assert "accessToken" not in blob
    assert "ssrToken" not in blob
    assert "remoteAddress" not in blob
    assert not re.search(r"ey[A-Za-z0-9_-]{20,}", blob)  # no JWT


def test_dto_detail_json_roundtrip():
    """detail survives model_dump(mode='json') -> validate, the exact transform
    _upsert_job / _job_dto_from_row use for the JSONB column."""
    from app.schemas import JobListingDTO, NormalizedJob

    nj = gn.extract_glints_job(_next_data())
    dto = JobListingDTO(
        id="1",
        portal="glints",
        title="T",
        company="C",
        company_logo_bg="#000",
        location="Jakarta",
        work_type="onsite",
        seniority="mid",
        salary_min=0,
        salary_max=0,
        posted_date="2026-01-01",
        posted_label="now",
        apply_url="https://x",
        match_score=None,
        cosine=0.0,
        llm_score=0,
        matched_skills=[],
        missing_skills=[],
        summary_id="",
        summary_en="",
        description="d",
        requirements="",
        detail=nj,
    )
    blob = dto.detail.model_dump(mode="json")
    rebuilt = NormalizedJob.model_validate(blob)
    assert rebuilt.company.name == "Nodewave Indonesia"
    assert rebuilt.title == nj.title
