from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import embeddings_service
from app.ai.llm import get_llm
from app.ai.matcher import cosine_score_pct, hybrid_score
from app.models import JobListing, MatchResult
from app.schemas import JobListingDTO
from app.services import cv_service
from app.services.search_service import _apply_jd_extraction, _job_dto_from_row


class JobNotFoundError(Exception):
    pass


class NoActiveCVError(Exception):
    pass


async def analyze_job(
    session: AsyncSession,
    user_id: int,
    job_id: int,
    *,
    force_refresh: bool = False,
) -> JobListingDTO:
    """Generate or return an on-demand AI match analysis for one job.

    Search remains a discovery flow; this service is the explicit Career
    Copilot analysis step triggered from a job detail action.
    """
    job = (
        await session.execute(select(JobListing).where(JobListing.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise JobNotFoundError(str(job_id))

    cv = await cv_service.get_active_cv_full(session, user_id)
    if cv is None:
        raise NoActiveCVError()

    existing = None
    if not force_refresh:
        existing = (
            await session.execute(
                select(MatchResult)
                .where(
                    MatchResult.user_id == user_id,
                    MatchResult.job_id == job.id,
                    MatchResult.cv_id == cv.id,
                )
                .order_by(MatchResult.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    if existing is not None:
        return _job_with_match(job, existing)

    dto = _job_dto_from_row(job)
    llm = get_llm()

    if not (
        dto.responsibilities
        or dto.mandatory_requirements
        or dto.nice_to_have_requirements
        or dto.skills_tags
        or dto.benefits
    ):
        [dto] = await _apply_jd_extraction(llm, [dto])
        _copy_structured_fields(job, dto)

    embedding = list(job.embedding) if job.embedding is not None else None
    if embedding is None:
        embeddings_service.load()
        [embedding] = await embeddings_service.encode([dto.description or dto.title])
        job.embedding = embedding

    cos = cosine_score_pct(cv.embedding, embedding)
    try:
        [out] = await llm.score_jobs(cv.text_content, [dto])
    except Exception as e:
        logger.warning(f"on-demand llm score failed: {e}; cosine-only fallback")
        out = type(
            "_CosineOnlyMatch",
            (),
            {
                "llm_score": cos,
                "matched_skills": [],
                "missing_skills": [],
                "summary_id": "Score dihitung dari cosine similarity karena AI provider sedang dibatasi.",
                "summary_en": "Score calculated from cosine similarity because the AI provider is rate-limited.",
            },
        )()

    final = hybrid_score(cos, out.llm_score)
    match = MatchResult(
        user_id=user_id,
        query_id=None,
        job_id=job.id,
        cv_id=cv.id,
        match_score=final,
        cosine_score=float(cos) / 100.0,
        llm_score=out.llm_score,
        matched_skills=out.matched_skills,
        missing_skills=out.missing_skills,
        summary_id=out.summary_id,
        summary_en=out.summary_en,
    )
    session.add(match)
    await session.flush()
    await session.refresh(match)
    return _job_with_match(job, match)


def _copy_structured_fields(job: JobListing, dto: JobListingDTO) -> None:
    job.responsibilities = dto.responsibilities or None
    job.mandatory_requirements = dto.mandatory_requirements or None
    job.nice_to_have_requirements = dto.nice_to_have_requirements or None
    job.skills_tags = dto.skills_tags or None
    job.benefits = dto.benefits or None


def _job_with_match(job: JobListing, match: MatchResult) -> JobListingDTO:
    return _job_dto_from_row(job).model_copy(
        update={
            "match_score": match.match_score,
            "cosine": float(match.cosine_score or 0.0),
            "llm_score": match.llm_score or 0,
            "matched_skills": list(match.matched_skills or []),
            "missing_skills": list(match.missing_skills or []),
            "summary_id": match.summary_id or "",
            "summary_en": match.summary_en or "",
        }
    )
