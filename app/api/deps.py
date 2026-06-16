from collections.abc import AsyncIterator

from fastapi import Cookie, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal
from app.models import UserAccount
from app.services import auth_service


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DBSession = Depends(get_db)


async def get_current_user(
    session: AsyncSession = Depends(get_db),
    token: str | None = Cookie(default=None, alias=auth_service.COOKIE_NAME),
) -> UserAccount:
    try:
        return await auth_service.get_user_by_session_token(session, token)
    except auth_service.InvalidSessionError as e:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "UNAUTHORIZED", "message": "Authentication required"}},
        ) from e
