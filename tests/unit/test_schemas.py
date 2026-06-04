from app.schemas import (
    JobListingDTO,
    PartialResultEvent,
)


def test_job_listing_dto_round_trip():
    raw = {
        "id": "1",
        "portal": "glints",
        "title": "Backend Engineer",
        "company": "Tokopedia",
        "company_logo_bg": "#abc",
        "location": "Jakarta",
        "work_type": "remote",
        "seniority": "mid",
        "salary_min": 12000000,
        "salary_max": 18000000,
        "posted_date": "2026-05-26",
        "posted_label": "2 days ago",
        "apply_url": "https://example.com",
        "match_score": 87,
        "cosine": 0.78,
        "llm_score": 88,
        "matched_skills": ["Node.js"],
        "missing_skills": ["Kubernetes"],
        "summary_id": "Cocok",
        "summary_en": "Good fit",
        "description": "...",
        "requirements": "...",
    }
    dto = JobListingDTO.model_validate(raw)
    assert dto.id == "1"
    assert dto.match_score == 87


def test_partial_result_event_shape():
    job = JobListingDTO(
        id="1",
        portal="glints",
        title="t",
        company="c",
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
    ev = PartialResultEvent(job=job)
    payload = ev.model_dump()
    assert payload["type"] == "partial_result"
    assert payload["job"]["match_score"] is None


def test_job_listing_dto_new_metadata_fields_default_to_empty():
    """New structured metadata fields must exist and default to []."""
    job = JobListingDTO(
        id="1",
        portal="glints",
        title="t",
        company="c",
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
    assert job.responsibilities == []
    assert job.mandatory_requirements == []
    assert job.nice_to_have_requirements == []
    assert job.skills_tags == []
    assert job.benefits == []
