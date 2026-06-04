import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.events import bus
from app.schemas import SearchAccepted, SearchRequest
from app.services import cv_service, search_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchAccepted, status_code=202)
async def start_search(
    req: SearchRequest, session: AsyncSession = Depends(get_db)
) -> SearchAccepted:
    # Spec §12: CV-missing must be surfaced synchronously as 409, not via WS
    active = await cv_service.get_active_cv(session)
    if active is None:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "NO_CV", "message": "Upload a CV before searching"}},
        )
    query_id = await search_service.create_search_row(session, req.query)
    await session.commit()
    bus.open(query_id)
    asyncio.create_task(search_service.run_pipeline(query_id, req.query, req.force_refresh))
    return SearchAccepted(query_id=query_id)


@router.get("/history")
async def list_history(limit: int = 50, session: AsyncSession = Depends(get_db)) -> list[dict]:
    return await search_service.get_history(session, limit=limit)


@router.get("/{query_id}")
async def get_one(query_id: int, session: AsyncSession = Depends(get_db)) -> dict:
    row = await search_service.get_search(session, query_id)
    if row is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "search not found"}}
        )
    return row
