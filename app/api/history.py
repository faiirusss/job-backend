from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import UserAccount
from app.services import search_service

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
async def history(
    limit: int = 50,
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> list[dict]:
    return await search_service.get_history(session, current_user.id, limit=limit)
