from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.services import search_service

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
async def history(limit: int = 50, session: AsyncSession = Depends(get_db)) -> list[dict]:
    return await search_service.get_history(session, limit=limit)
