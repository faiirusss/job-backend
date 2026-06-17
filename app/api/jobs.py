from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import JobListing, MatchResult, UserAccount
from app.schemas import CoverLetterRequest, CoverLetterResponse, JobListingDTO, MatchScoreRequest
from app.services import cover_letter_service, job_detail_service, match_service
from app.services.cover_letter_service import (
    CoverLetterGenerationError,
    JobNotFoundError,
    NoActiveCVError,
)
from app.services.match_service import (
    JobNotFoundError as MatchJobNotFoundError,
    NoActiveCVError as MatchNoActiveCVError,
)
from app.services.search_service import _job_dto_from_row

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(
    job_id: int,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> dict:
    job = (
        await session.execute(select(JobListing).where(JobListing.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "job not found"}}
        )
    await job_detail_service.ensure_job_detail(session, job)
    match = (
        await session.execute(
            select(MatchResult)
            .where(MatchResult.user_id == current_user.id, MatchResult.job_id == job_id)
            .order_by(MatchResult.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    dto = _job_dto_from_row(job)
    if match and match_service.is_match_current(match, job):
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


@router.post("/{job_id}/match-score", response_model=JobListingDTO)
async def match_score(
    job_id: int,
    req: MatchScoreRequest = MatchScoreRequest(),
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> JobListingDTO:
    try:
        return await match_service.analyze_job(
            session,
            current_user.id,
            job_id,
            force_refresh=req.force_refresh,
        )
    except MatchJobNotFoundError as e:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "job not found"}}
        ) from e
    except MatchNoActiveCVError as e:
        raise HTTPException(
            status_code=409, detail={"error": {"code": "NO_CV", "message": "No active CV"}}
        ) from e


@router.post("/{job_id}/cover-letter", response_model=CoverLetterResponse)
async def cover_letter(
    job_id: int,
    _req: CoverLetterRequest = CoverLetterRequest(),
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> CoverLetterResponse:
    try:
        return await cover_letter_service.generate(session, current_user.id, job_id)
    except JobNotFoundError as e:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "job not found"}}
        ) from e
    except NoActiveCVError as e:
        raise HTTPException(
            status_code=409, detail={"error": {"code": "NO_CV", "message": "No active CV"}}
        ) from e
    except CoverLetterGenerationError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "COVER_LETTER_FAILED",
                    "message": "Cover letter belum bisa dibuat. Coba lagi sebentar lagi.",
                }
            },
        ) from e
