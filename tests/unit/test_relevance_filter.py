from app.schemas import JobListingDTO, SearchParams
from app.services.relevance_filter import filter_relevant_jobs


def _job(
    job_id: str,
    *,
    title: str,
    location: str = "Jakarta",
    description: str = "",
    skills: list[str] | None = None,
    work_type: str = "onsite",
) -> JobListingDTO:
    return JobListingDTO(
        id=job_id,
        portal="glints",
        title=title,
        company="Acme",
        company_logo_bg="#000",
        location=location,
        work_type=work_type,  # type: ignore[arg-type]
        seniority="mid",
        salary_min=0,
        salary_max=0,
        posted_date="2026-01-01",
        posted_label="recent",
        apply_url="https://example.com",
        match_score=None,
        cosine=0.0,
        llm_score=0,
        matched_skills=[],
        missing_skills=[],
        summary_id="",
        summary_en="",
        description=description,
        requirements="",
        skills_tags=skills or [],
    )


def test_filters_role_and_location_for_laravel_jakarta_query():
    jobs = [
        _job("ok", title="PHP Developer", location="Jakarta Barat", skills=["Laravel", "MySQL"]),
        _job("wrong-role", title="Data Analyst", location="Jakarta", skills=["SQL", "Tableau"]),
        _job("wrong-location", title="Laravel Developer", location="Bandung", skills=["Laravel"]),
    ]
    params = SearchParams(role_keywords=["laravel"], location=["Jakarta"])

    kept, stats = filter_relevant_jobs(jobs, params)

    assert [job.id for job in kept] == ["ok"]
    assert stats.dropped_role == 1
    assert stats.dropped_location == 1


def test_ignores_chat_filler_and_location_tokens_in_role_keywords():
    jobs = [
        _job("ok", title="Laravel Engineer", location="Jakarta"),
        _job("bad", title="Backend Engineer", location="Jakarta", skills=["Python"]),
    ]
    params = SearchParams(
        role_keywords=["tolong", "kerjaan", "laravel", "jakarta"],
        location=["Jakarta"],
    )

    kept, stats = filter_relevant_jobs(jobs, params)

    assert [job.id for job in kept] == ["ok"]
    assert stats.dropped_role == 1


def test_does_not_location_filter_indonesia_only_search():
    jobs = [
        _job("jakarta", title="Laravel Developer", location="Jakarta", skills=["Laravel"]),
        _job("bandung", title="Laravel Developer", location="Bandung", skills=["Laravel"]),
    ]
    params = SearchParams(role_keywords=["laravel"], location=["Indonesia"])

    kept, stats = filter_relevant_jobs(jobs, params)

    assert {job.id for job in kept} == {"jakarta", "bandung"}
    assert stats.dropped_location == 0


def test_dki_jakarta_location_matches_jakarta_area_jobs():
    jobs = [
        _job("jakarta", title="Laravel Developer", location="Jakarta Barat", skills=["Laravel"]),
        _job("bandung", title="Laravel Developer", location="Bandung", skills=["Laravel"]),
    ]
    params = SearchParams(role_keywords=["laravel"], location=["DKI Jakarta"])

    kept, stats = filter_relevant_jobs(jobs, params)

    assert [job.id for job in kept] == ["jakarta"]
    assert stats.dropped_location == 1


def test_filters_explicit_work_type():
    jobs = [
        _job("remote", title="Laravel Developer", work_type="remote", skills=["Laravel"]),
        _job("onsite", title="Laravel Developer", work_type="onsite", skills=["Laravel"]),
    ]
    params = SearchParams(role_keywords=["laravel"], work_type=["remote"])

    kept, stats = filter_relevant_jobs(jobs, params)

    assert [job.id for job in kept] == ["remote"]
    assert stats.dropped_work_type == 1


def test_generic_role_terms_do_not_match_description_only_mentions():
    jobs = [
        _job(
            "passing-mention",
            title="People Operations Manager",
            description="Collaborate with engineering teams on hiring plans.",
        ),
        _job("title-match", title="Software Engineer"),
    ]
    params = SearchParams(role_keywords=["engineer"])

    kept, stats = filter_relevant_jobs(jobs, params)

    assert [job.id for job in kept] == ["title-match"]
    assert stats.dropped_role == 1


def test_role_matching_uses_exact_tokens_not_substrings():
    jobs = [
        _job("java", title="Java Developer"),
        _job("javascript", title="JavaScript Developer"),
    ]
    params = SearchParams(role_keywords=["java"])

    kept, stats = filter_relevant_jobs(jobs, params)

    assert [job.id for job in kept] == ["java"]
    assert stats.dropped_role == 1
