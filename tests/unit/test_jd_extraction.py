import pytest

from app.ai.fake_llm import FakeLLM
from app.ai.llm import JDExtraction
from app.schemas import JobListingDTO
from app.services.search_service import _apply_jd_extraction, _dedupe_preserve


def _job(jid: str, desc: str, *, skills=None, benefits=None) -> JobListingDTO:
    return JobListingDTO(
        id=jid,
        portal="glints",
        title="Backend Engineer",
        company="Acme",
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
        skills_tags=skills or [],
        benefits=benefits or [],
    )


def test_dedupe_preserve_unions_case_insensitively_keeping_order():
    out = _dedupe_preserve(["Python", "Docker"], ["docker", "AWS", ""])
    assert out == ["Python", "Docker", "AWS"]


@pytest.mark.asyncio
async def test_apply_jd_extraction_fills_prose_and_unions_skills_benefits():
    """Prose fields come from the LLM; skills_tags/benefits union the scraper's
    structured arrays with the LLM's reading (no data loss, deduped)."""
    desc = (
        "Membangun API dengan FastAPI\n"
        "Kualifikasi:\n"
        "Pengalaman Python minimal 3 tahun\n"
        "Benefit\n"
        "Asuransi kesehatan untuk seluruh keluarga\n"
    )
    job = _job("j1", desc, skills=["Python"], benefits=["BPJS"])
    out = await _apply_jd_extraction(FakeLLM(), [job])
    assert len(out) == 1
    j = out[0]
    assert any("FastAPI" in r for r in j.responsibilities)
    assert any("Python minimal 3 tahun" in m for m in j.mandatory_requirements)
    # Structured array value ("Python"/"BPJS") preserved; LLM additions unioned in.
    assert "Python" in j.skills_tags
    assert "BPJS" in j.benefits
    assert any("Asuransi" in b for b in j.benefits)
    # Raw description block is untouched.
    assert j.description == desc


@pytest.mark.asyncio
async def test_apply_jd_extraction_falls_back_on_llm_error():
    """If extraction raises, jobs are returned with descriptions intact."""

    class _BoomLLM(FakeLLM):
        async def extract_jd_fields(self, jobs):
            raise RuntimeError("gemini down")

    job = _job("j1", "some description", skills=["Go"])
    out = await _apply_jd_extraction(_BoomLLM(), [job])
    assert out == [job]
    assert out[0].description == "some description"
    assert out[0].skills_tags == ["Go"]


@pytest.mark.asyncio
async def test_apply_jd_extraction_handles_short_llm_result():
    """A short LLM result leaves trailing jobs unchanged rather than dropping them."""

    class _ShortLLM(FakeLLM):
        async def extract_jd_fields(self, jobs):
            return [JDExtraction([], [], [], [], [])]  # only one result for N jobs

    jobs = [_job("j1", "d1"), _job("j2", "d2")]
    out = await _apply_jd_extraction(_ShortLLM(), jobs)
    assert len(out) == 2
    assert out[1].id == "j2"
