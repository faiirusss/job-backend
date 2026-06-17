from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm import get_llm
from app.models import CoverLetter, JobListing, MatchResult
from app.schemas import CoverLetterResponse
from app.services import cv_service, job_detail_service
from app.services.search_service import _job_dto_from_row


class JobNotFoundError(Exception):
    pass


class NoActiveCVError(Exception):
    pass


async def generate(session: AsyncSession, user_id: int, job_id: int) -> CoverLetterResponse:
    job = (
        await session.execute(select(JobListing).where(JobListing.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise JobNotFoundError(str(job_id))

    cv = await cv_service.get_active_cv_full(session, user_id)
    if cv is None:
        raise NoActiveCVError()

    await job_detail_service.ensure_job_detail(session, job)

    existing = (
        await session.execute(
            select(CoverLetter).where(CoverLetter.job_id == job.id, CoverLetter.cv_id == cv.id)
        )
    ).scalar_one_or_none()
    if existing is not None and _cover_letter_is_current(existing, job):
        return CoverLetterResponse(
            content_id=existing.content_id,
            content_en=existing.content_en,
            word_count_id=existing.word_count_id or len(existing.content_id.split()),
            word_count_en=existing.word_count_en or len(existing.content_en.split()),
            from_cache=True,
            generated_at=existing.generated_at,
        )

    job_dto = _job_dto_from_row(job)
    match = (
        await session.execute(
            select(MatchResult)
            .where(
                MatchResult.user_id == user_id,
                MatchResult.job_id == job.id,
                MatchResult.cv_id == cv.id,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    matched_skills = list(match.matched_skills) if match else []

    pair = await get_llm().generate_cover_letter(cv.text_content, job_dto, matched_skills)
    generated_at = datetime.now(UTC)
    if existing is not None:
        existing.content_id = pair.content_id
        existing.content_en = pair.content_en
        existing.word_count_id = pair.word_count_id
        existing.word_count_en = pair.word_count_en
        existing.generated_at = generated_at
    else:
        row = CoverLetter(
            user_id=user_id,
            job_id=job.id,
            cv_id=cv.id,
            content_id=pair.content_id,
            content_en=pair.content_en,
            word_count_id=pair.word_count_id,
            word_count_en=pair.word_count_en,
            generated_at=generated_at,
        )
        session.add(row)
    await session.flush()
    return CoverLetterResponse(
        content_id=pair.content_id,
        content_en=pair.content_en,
        word_count_id=pair.word_count_id,
        word_count_en=pair.word_count_en,
        from_cache=False,
        generated_at=generated_at,
    )


def _cover_letter_is_current(row: CoverLetter, job: JobListing) -> bool:
    if row.generated_at is None or job.scraped_at is None:
        return True
    try:
        return row.generated_at >= job.scraped_at
    except TypeError:
        return row.generated_at.replace(tzinfo=None) >= job.scraped_at.replace(tzinfo=None)
