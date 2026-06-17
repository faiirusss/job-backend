from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import embeddings_service
from app.ai.llm import CoverLetterPair
from app.models import CoverLetter, JobListing
from app.services import cover_letter_service, cv_service

PDF = Path(__file__).parent.parent / "fixtures" / "sample_cv.pdf"


class CoverLetterLLM:
    async def generate_cover_letter(self, cv_text, job, matched_skills):
        content_id = (
            f"Yth. Tim HRD {job.company}, dengan hormat saya bermaksud melamar posisi "
            f"{job.title}. Selama lima tahun terakhir saya membangun layanan backend yang "
            "andal menggunakan Python dan FastAPI, merancang basis data PostgreSQL yang "
            "efisien, serta menerapkan praktik pengujian otomatis. Saya yakin pengalaman ini "
            "relevan dengan kebutuhan tim Anda dan siap memberikan kontribusi nyata. Saya "
            "sangat antusias mendiskusikan bagaimana keahlian saya mendukung tujuan "
            "perusahaan. Terima kasih atas perhatian dan kesempatan yang diberikan."
        )
        content_en = (
            f"Dear Hiring Manager, I am writing to apply for the {job.title} position at "
            f"{job.company}. Over the past five years I have built reliable backend services "
            "with Python and FastAPI, designed efficient PostgreSQL databases, and established "
            "automated testing practices that improved delivery speed. I am confident this "
            "experience aligns closely with your team's needs and that I can contribute "
            "meaningfully from day one. I would welcome the opportunity to discuss how my "
            "skills support your goals. Thank you for your time and consideration."
        )
        return CoverLetterPair(
            content_id=content_id,
            content_en=content_en,
            word_count_id=len(content_id.split()),
            word_count_en=len(content_en.split()),
        )


async def _seed_job(db_session: AsyncSession) -> int:
    job = JobListing(
        external_id="g-test",
        portal="glints",
        title="Backend Engineer",
        company="Tokopedia",
        apply_url="https://example.com",
        description="Need Python FastAPI",
    )
    db_session.add(job)
    await db_session.flush()
    await db_session.refresh(job)
    return job.id


async def test_generate_then_cache_hit(
    db_session: AsyncSession, test_user_id: int, monkeypatch
):
    embeddings_service.load()
    await cv_service.upload_cv(db_session, test_user_id, "cv.pdf", PDF.read_bytes())
    await db_session.commit()
    job_id = await _seed_job(db_session)
    await db_session.commit()
    monkeypatch.setattr(cover_letter_service, "_get_cover_letter_llm", lambda: CoverLetterLLM())

    out1 = await cover_letter_service.generate(db_session, test_user_id, job_id)
    await db_session.commit()
    assert out1.from_cache is False
    assert "Tokopedia" in out1.content_id

    out2 = await cover_letter_service.generate(db_session, test_user_id, job_id)
    assert out2.from_cache is True
    assert out2.content_id == out1.content_id


async def test_generate_raises_generation_error_when_qwen_fails(
    db_session: AsyncSession, test_user_id: int, monkeypatch
):
    class FailingLLM:
        async def generate_cover_letter(self, cv_text, job, matched_skills):
            raise RuntimeError("429 quota exceeded")

    embeddings_service.load()
    await cv_service.upload_cv(db_session, test_user_id, "cv.pdf", PDF.read_bytes())
    await db_session.commit()
    job_id = await _seed_job(db_session)
    await db_session.commit()
    monkeypatch.setattr(cover_letter_service, "_get_cover_letter_llm", lambda: FailingLLM())

    with pytest.raises(cover_letter_service.CoverLetterGenerationError):
        await cover_letter_service.generate(db_session, test_user_id, job_id)

    cached = (
        await db_session.execute(select(CoverLetter).where(CoverLetter.job_id == job_id))
    ).scalar_one_or_none()
    assert cached is None


async def test_generate_raises_generation_error_when_qwen_returns_empty(
    db_session: AsyncSession, test_user_id: int, monkeypatch
):
    class EmptyLLM:
        async def generate_cover_letter(self, cv_text, job, matched_skills):
            return CoverLetterPair("", "", 0, 0)

    embeddings_service.load()
    await cv_service.upload_cv(db_session, test_user_id, "cv.pdf", PDF.read_bytes())
    await db_session.commit()
    job_id = await _seed_job(db_session)
    await db_session.commit()
    monkeypatch.setattr(cover_letter_service, "_get_cover_letter_llm", lambda: EmptyLLM())

    with pytest.raises(cover_letter_service.CoverLetterGenerationError):
        await cover_letter_service.generate(db_session, test_user_id, job_id)
