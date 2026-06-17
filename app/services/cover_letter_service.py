from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm import LLM, CoverLetterPair
from app.config import settings
from app.models import CoverLetter, JobListing, MatchResult
from app.schemas import CoverLetterResponse, JobListingDTO
from app.services import cv_service, job_detail_service
from app.services.search_service import _job_dto_from_row

_MIN_COVER_LETTER_WORDS = 50


class JobNotFoundError(Exception):
    pass


class NoActiveCVError(Exception):
    pass


class CoverLetterGenerationError(Exception):
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
            generated_at=existing.generated_at or datetime.now(UTC),
        )

    try:
        job_dto = _job_dto_from_row(job)
    except Exception as exc:
        logger.warning("cover letter job DTO build failed for job {}: {}", job.id, exc)
        raise CoverLetterGenerationError("Job data is not valid for cover letter") from exc

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
    matched_skills = list(match.matched_skills or []) if match else []

    pair = await _generate_pair(cv.text_content, job_dto, matched_skills)
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


async def _generate_pair(
    cv_text: str, job: JobListingDTO, matched_skills: list[str]
) -> CoverLetterPair:
    try:
        pair = await _get_cover_letter_llm().generate_cover_letter(cv_text, job, matched_skills)
    except Exception as exc:
        logger.warning("qwen cover letter generation failed: {}", exc)
        raise CoverLetterGenerationError("Qwen cover letter generation failed") from exc

    if _cover_letter_pair_has_content(pair):
        return pair

    logger.warning(
        "qwen cover letter generation returned incomplete content: id_words={}, en_words={}",
        _word_count(pair.content_id),
        _word_count(pair.content_en),
    )
    raise CoverLetterGenerationError("Qwen returned an incomplete cover letter")


def _get_cover_letter_llm() -> LLM:
    if not settings.qwen_api_key:
        raise CoverLetterGenerationError("QWEN_API_KEY must be set for cover letter generation")

    from app.ai.qwen import QwenLLM

    return QwenLLM(
        api_key=settings.qwen_api_key,
        model=settings.resolved_cover_letter_qwen_model,
        base_url=settings.qwen_base_url,
        rpm=settings.qwen_rpm_limit,
        enable_thinking=settings.qwen_enable_thinking,
    )


def _cover_letter_pair_has_content(pair: CoverLetterPair) -> bool:
    return (
        _word_count(pair.content_id) >= _MIN_COVER_LETTER_WORDS
        and _word_count(pair.content_en) >= _MIN_COVER_LETTER_WORDS
    )


def _word_count(text: str) -> int:
    return len(text.strip().split()) if text.strip() else 0


def _cover_letter_is_current(row: CoverLetter, job: JobListing) -> bool:
    if row.generated_at is None or job.scraped_at is None:
        return True
    try:
        return row.generated_at >= job.scraped_at
    except TypeError:
        return row.generated_at.replace(tzinfo=None) >= job.scraped_at.replace(tzinfo=None)
