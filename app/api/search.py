import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.events import bus
from app.models import UserAccount
from app.schemas import SearchAccepted, SearchRequest, SearchResultsResponse
from app.services import cv_service, search_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchAccepted, status_code=202)
async def start_search(
    req: SearchRequest,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> SearchAccepted:
    active = await cv_service.get_active_cv(session, current_user.id)
    if active is None:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "NO_CV", "message": "Upload a CV before searching"}},
        )
    query_id = await search_service.create_search_row(session, req.query, user_id=current_user.id)
    await session.commit()
    bus.open(query_id)
    asyncio.create_task(search_service.run_pipeline(query_id, req.query, req.force_refresh))
    return SearchAccepted(query_id=query_id)


@router.get("/history")
async def list_history(
    limit: int = 50,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> list[dict]:
    return await search_service.get_history(session, current_user.id, limit=limit)


@router.get("/{query_id}/results", response_model=SearchResultsResponse)
async def get_results(
    query_id: int,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> SearchResultsResponse:
    row = await search_service.get_search_results(session, current_user.id, query_id)
    if row is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "search not found"}}
        )
    return row


@router.get("/{query_id}")
async def get_one(
    query_id: int,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> dict:
    row = await search_service.get_search(session, current_user.id, query_id)
    if row is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "search not found"}}
        )
    return row
