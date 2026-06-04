import pytest

from app.ai.fake_llm import FakeLLM
from app.schemas import JobListingDTO, SearchParams


def _job(jid: str, title="Backend Engineer", company="Acme", desc="Python FastAPI"):
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
        description=desc,
        requirements="",
    )


@pytest.mark.asyncio
async def test_parse_intent_detects_remote_and_salary():
    llm = FakeLLM()
    params = await llm.parse_intent("Cari loker Node.js remote, gaji minimal 15 juta")
    assert "remote" in params.work_type
    assert params.salary_min_idr == 15_000_000


@pytest.mark.asyncio
async def test_parse_intent_extracts_role_keywords():
    llm = FakeLLM()
    params = await llm.parse_intent("Senior python backend engineer in Jakarta")
    assert any(k.lower() == "python" for k in params.role_keywords)


@pytest.mark.asyncio
async def test_score_jobs_returns_deterministic_results():
    llm = FakeLLM()
    cv_text = "Experienced Python FastAPI engineer with PostgreSQL and Docker."
    job = _job("j1", desc="Python FastAPI PostgreSQL Docker Kubernetes")
    out1 = await llm.score_jobs(cv_text, [job])
    out2 = await llm.score_jobs(cv_text, [job])
    assert out1[0].llm_score == out2[0].llm_score
    assert 60 <= out1[0].llm_score <= 89
    assert out1[0].matched_skills  # non-empty


@pytest.mark.asyncio
async def test_generate_cover_letter_substitutes_company_and_title():
    llm = FakeLLM()
    job = _job("j1", title="Senior Backend", company="Tokopedia")
    pair = await llm.generate_cover_letter("My CV text", job, ["Python"])
    assert "Tokopedia" in pair.content_id
    assert "Tokopedia" in pair.content_en
    assert "Senior Backend" in pair.content_id
    assert pair.word_count_id > 100


@pytest.mark.asyncio
async def test_generate_intro_mentions_role_and_location():
    llm = FakeLLM()
    params = SearchParams(
        role_keywords=["React", "Junior"],
        location=["Jakarta"],
        work_type=["remote"],
    )
    msg = await llm.generate_intro("Cari React Junior di Jakarta remote", params)
    assert "React" in msg
    assert "Jakarta" in msg
    assert "Glints" in msg
    assert len(msg) < 200


def test_factory_returns_fake_llm_when_flag_true(monkeypatch):
    monkeypatch.setenv("USE_FAKE_LLM", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from importlib import reload

    import app.config as cfg

    reload(cfg)
    import app.ai.llm as llm_mod

    reload(llm_mod)
    from app.ai.fake_llm import FakeLLM

    assert isinstance(llm_mod.get_llm(), FakeLLM)


@pytest.mark.asyncio
async def test_extract_jd_fields_buckets_description_lines():
    """The FakeLLM stand-in buckets a free-form description block by keyword cue,
    drops heading-only lines, and returns one JDExtraction per job in order."""
    llm = FakeLLM()
    desc = (
        "Tentang Peran\n"
        "Membangun API dengan FastAPI\n"
        "Menulis unit test\n"
        "Kualifikasi:\n"
        "Pengalaman Python minimal 3 tahun\n"
        "Nilai Plus\n"
        "Pengalaman dengan AWS preferred\n"
        "Benefit\n"
        "Asuransi kesehatan untuk keluarga\n"
    )
    job = _job("j-extract", desc=desc)
    out = await llm.extract_jd_fields([job])
    assert len(out) == 1
    ext = out[0]
    assert "Membangun API dengan FastAPI" in ext.responsibilities
    assert "Menulis unit test" in ext.responsibilities
    assert any("Pengalaman Python minimal 3 tahun" in m for m in ext.mandatory_requirements)
    assert any("AWS" in n for n in ext.nice_to_have_requirements)
    assert any("Asuransi" in b for b in ext.benefits)
    # "Kualifikasi:", "Nilai Plus", "Benefit" are heading-only -> dropped, not items.
    assert "Kualifikasi:" not in ext.mandatory_requirements
    assert "Benefit" not in ext.benefits


@pytest.mark.asyncio
async def test_extract_jd_fields_empty_description_returns_empty_lists():
    llm = FakeLLM()
    out = await llm.extract_jd_fields([_job("j-empty", desc="")])
    assert len(out) == 1
    ext = out[0]
    assert ext.responsibilities == []
    assert ext.mandatory_requirements == []
    assert ext.benefits == []


@pytest.mark.asyncio
async def test_score_jobs_uses_structured_fields_for_matching():
    """FakeLLM must draw matched/missing from structured fields, not just description."""
    llm = FakeLLM()
    cv_text = "I have extensive Kubernetes and Docker experience."
    job = _job("j-struct", desc="")
    job = job.model_copy(
        update={
            "mandatory_requirements": ["Kubernetes expertise required", "Docker containerization"],
            "skills_tags": ["Kubernetes", "Docker", "Python"],
        }
    )
    results = await llm.score_jobs(cv_text, [job])
    assert len(results) == 1
    all_skills = results[0].matched_skills + results[0].missing_skills
    assert any("kubernetes" in s or "docker" in s for s in all_skills), (
        f"Expected kubernetes/docker in skills, got matched={results[0].matched_skills} "
        f"missing={results[0].missing_skills}"
    )
