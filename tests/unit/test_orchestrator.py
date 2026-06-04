from app.schemas import JobListingDTO
from app.scrapers.orchestrator import dedupe_by_company_title


def _job(jid: str, title: str, company: str) -> JobListingDTO:
    return JobListingDTO(
        id=jid,
        portal="glints",
        title=title,
        company=company,
        company_logo_bg="#000",
        location="Jakarta",
        work_type="remote",
        seniority="mid",
        salary_min=0,
        salary_max=0,
        posted_date="2026-01-01",
        posted_label="now",
        apply_url="https://example.com",
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


def test_dedupe_drops_same_company_title_case_insensitive():
    jobs = [
        _job("1", "Backend Engineer", "Tokopedia"),
        _job("2", "  backend engineer  ", "TOKOPEDIA"),
        _job("3", "Frontend Engineer", "Tokopedia"),
    ]
    out = dedupe_by_company_title(jobs)
    ids = [j.id for j in out]
    assert "1" in ids and "3" in ids
    assert "2" not in ids
    assert len(out) == 2


def test_registry_includes_linkedin():
    from app.scrapers.linkedin import LinkedInScraper
    from app.scrapers.orchestrator import _REGISTRY

    assert _REGISTRY.get("linkedin") is LinkedInScraper
