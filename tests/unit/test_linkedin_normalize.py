import pathlib

from app.scrapers import linkedin_normalize as ln

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


def _listing_html() -> str:
    return (FIXTURES / "linkedin_listing.html").read_text(encoding="utf-8")


def _detail_html() -> str:
    return (FIXTURES / "linkedin_detail.html").read_text(encoding="utf-8")


def test_with_start_sets_pagination_offset():
    url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=python&start=0"
    assert ln.with_start(url, 25).endswith("start=25")
    assert "keywords=python" in ln.with_start(url, 25)


def test_sanitize_html_strips_attrs_and_disallowed_tags():
    raw = '<p class="x">hi</p><span style="color:red">drop wrapper keep text</span><script>bad()</script>'
    out = ln.sanitize_html(raw)
    assert "<p>hi</p>" in out
    assert "drop wrapper keep text" in out  # span unwrapped, text kept
    assert "<span" not in out
    assert "class=" not in out
    assert "script" not in out and "bad()" not in out  # script fully removed


def test_parse_listing_cards_extracts_two_jobs():
    jobs = ln.parse_listing_cards(_listing_html())
    assert len(jobs) == 2
    a, b = jobs
    assert a.id == "3901234567"
    assert a.portal == "linkedin"
    assert a.title == "Senior Backend Engineer"
    assert a.company == "Acme Corp"
    assert a.location == "Jakarta, Jakarta, Indonesia"
    assert (
        a.apply_url
        == "https://www.linkedin.com/jobs/view/senior-backend-engineer-at-acme-3901234567"
    )
    assert a.posted_date == "2026-05-20"
    assert a.work_type == "onsite"
    assert b.id == "3907654321"
    assert b.work_type == "remote"  # location == "Remote"


def test_parse_listing_cards_extracts_logo_from_listing():
    # The listing card already carries the company logo (lazy-loaded in
    # data-delayed-url), so a job shows a logo without needing detail enrichment.
    a, b = ln.parse_listing_cards(_listing_html())
    assert a.company_logo_url == "https://media.licdn.com/dms/image/acme-logo.png"
    assert b.company_logo_url is None  # Globex card has no <img>


def test_parse_listing_cards_empty_html_returns_empty():
    assert ln.parse_listing_cards("") == []


def test_parse_listing_cards_falls_back_when_href_and_datetime_missing():
    html = (
        '<div data-entity-urn="urn:li:jobPosting:42">'
        '<a class="base-card__full-link"><span>x</span></a>'
        '<h3 class="base-search-card__title">QA Engineer</h3>'
        "<time>just now</time>"
        "</div>"
    )
    jobs = ln.parse_listing_cards(html)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.apply_url == "https://www.linkedin.com/jobs/view/42"
    assert j.posted_date == "2026-01-01"
    assert j.posted_label == "recent"


def test_extract_linkedin_job_maps_detail_fields():
    nj = ln.extract_linkedin_job(_detail_html(), job_id="3901234567")
    assert nj is not None
    assert nj.id == "3901234567"
    assert nj.title == "Senior Backend Engineer"
    assert nj.canonical_url == "https://www.linkedin.com/jobs/view/3901234567"
    # description_html: sanitized, span unwrapped, script gone, structure kept
    assert "<p>We are looking for a backend engineer" in nj.description_html
    assert "<li>Design and ship REST APIs</li>" in nj.description_html
    assert "inline span text" in nj.description_html
    assert "<span" not in nj.description_html
    assert "script" not in nj.description_html
    assert nj.requirements_html is None
    # employment type "Full-time" → FULL_TIME → Indonesian label
    assert nj.job_type == "FULL_TIME"
    assert nj.job_type_label == "Penuh Waktu"
    # job function / industries → category
    assert nj.category.name == "Engineering and Information Technology"
    assert "Software Development" in nj.category.breadcrumb
    # company
    assert nj.company.name == "Acme Corp"
    assert nj.company.website == "https://www.linkedin.com/company/acme"
    assert nj.company.logo_url == "https://media.licdn.com/dms/image/acme-logo.png"
    # salary absent → not shown
    assert nj.salary.show is False


def test_extract_linkedin_job_populates_location_and_applicants():
    nj = ln.extract_linkedin_job(_detail_html(), job_id="3901234567")
    assert nj is not None
    # location: top-card "Jakarta, Indonesia" → normalized
    assert nj.location.name == "Jakarta, Indonesia"
    assert nj.location.city == "Jakarta"
    assert nj.location.country == "Indonesia"
    # applicants: "Over 200 applicants" → 200
    assert nj.applicants_count == 200


def test_normalize_location_comma_split():
    loc = ln.normalize_location("Jakarta, Indonesia")
    assert loc.name == "Jakarta, Indonesia"
    assert loc.city == "Jakarta"
    assert loc.country == "Indonesia"
    assert loc.province is None


def test_normalize_location_static_map_metropolitan_area():
    loc = ln.normalize_location("Jakarta Metropolitan Area")
    assert loc.city == "Jakarta"
    assert loc.province == "DKI Jakarta"
    assert loc.country == "Indonesia"


def test_normalize_location_single_token_defaults_country():
    loc = ln.normalize_location("Bandung")
    assert loc.city == "Bandung"
    assert loc.country == "Indonesia"


def test_normalize_location_empty_is_blank():
    assert ln.normalize_location("").name == ""
    assert ln.normalize_location("   ").city is None


def test_applicants_count_parses_plain_and_over_forms():
    over = (
        '<div class="show-more-less-html__markup"><p>x</p></div>'
        '<span class="num-applicants__caption">56 applicants</span>'
    )
    assert ln.extract_linkedin_job(over, job_id="1").applicants_count == 56
    none = '<div class="show-more-less-html__markup"><p>x</p></div>'
    assert ln.extract_linkedin_job(none, job_id="1").applicants_count is None


def test_extract_linkedin_job_returns_none_without_markup():
    assert ln.extract_linkedin_job("<section><p>no markup div</p></section>") is None


def test_parse_job_detail_returns_detail_seniority_and_plaintext():
    parsed = ln.parse_job_detail(_detail_html(), job_id="3901234567")
    assert parsed["detail"] is not None
    assert parsed["seniority"] == "senior"  # "Mid-Senior level"
    assert "backend engineer" in parsed["description"].lower()
    assert "<" not in parsed["description"]  # plain text, not HTML


def test_parse_job_detail_empty_on_missing_markup():
    assert ln.parse_job_detail("<section></section>", job_id="x") == {}
