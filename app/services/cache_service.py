from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import CacheEntry


async def write(
    session: AsyncSession, params_hash: str, payload: dict[str, Any], ttl_seconds: int
) -> None:
    expires = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
    stmt = (
        insert(CacheEntry)
        .values(params_hash=params_hash, result_payload=payload, expires_at=expires)
        .on_conflict_do_update(
            index_elements=[CacheEntry.params_hash],
            set_={
                "result_payload": payload,
                "expires_at": expires,
                "created_at": datetime.now(UTC),
            },
        )
    )
    await session.execute(stmt)


async def lookup(session: AsyncSession, params_hash: str) -> dict[str, Any] | None:
    now = datetime.now(UTC)
    stmt = select(CacheEntry).where(
        CacheEntry.params_hash == params_hash,
        CacheEntry.expires_at > now,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    return row.result_payload if row else None


async def purge_expired(session: AsyncSession) -> int:
    now = datetime.now(UTC)
    result = await session.execute(delete(CacheEntry).where(CacheEntry.expires_at <= now))
    return result.rowcount or 0  # type: ignore[attr-defined]
