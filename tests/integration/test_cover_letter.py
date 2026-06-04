from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import embeddings_service
from app.models import JobListing
from app.services import cover_letter_service, cv_service

PDF = Path(__file__).parent.parent / "fixtures" / "sample_cv.pdf"


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


async def test_generate_then_cache_hit(db_session: AsyncSession):
    embeddings_service.load()
    await cv_service.upload_cv(db_session, "cv.pdf", PDF.read_bytes())
    await db_session.commit()
    job_id = await _seed_job(db_session)
    await db_session.commit()

    out1 = await cover_letter_service.generate(db_session, job_id)
    await db_session.commit()
    assert out1.from_cache is False
    assert "Tokopedia" in out1.content_id

    out2 = await cover_letter_service.generate(db_session, job_id)
    assert out2.from_cache is True
    assert out2.content_id == out1.content_id
