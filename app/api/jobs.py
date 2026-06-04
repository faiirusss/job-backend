from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models import JobListing, MatchResult
from app.schemas import CoverLetterRequest, CoverLetterResponse
from app.services import cover_letter_service
from app.services.cover_letter_service import JobNotFoundError, NoActiveCVError
from app.services.search_service import _job_dto_from_row

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(job_id: int, session: AsyncSession = Depends(get_db)) -> dict:
    job = (
        await session.execute(select(JobListing).where(JobListing.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "job not found"}}
        )
    match = (
        await session.execute(
            select(MatchResult)
            .where(MatchResult.job_id == job_id)
            .order_by(MatchResult.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    dto = _job_dto_from_row(job)
    if match:
        dto = dto.model_copy(
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
    return dto.model_dump()


@router.post("/{job_id}/cover-letter", response_model=CoverLetterResponse)
async def cover_letter(
    job_id: int,
    _req: CoverLetterRequest = CoverLetterRequest(),
    session: AsyncSession = Depends(get_db),
) -> CoverLetterResponse:
    try:
        return await cover_letter_service.generate(session, job_id)
    except JobNotFoundError as e:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "job not found"}}
        ) from e
    except NoActiveCVError as e:
        raise HTTPException(
            status_code=409, detail={"error": {"code": "NO_CV", "message": "No active CV"}}
        ) from e
