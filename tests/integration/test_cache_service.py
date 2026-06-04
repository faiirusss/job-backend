from sqlalchemy.ext.asyncio import AsyncSession

from app.services import cache_service


async def test_write_and_lookup_roundtrip(db_session: AsyncSession):
    await cache_service.write(db_session, "abc123", {"jobs": [{"id": "1"}]}, ttl_seconds=60)
    await db_session.commit()
    hit = await cache_service.lookup(db_session, "abc123")
    assert hit is not None
    assert hit["jobs"][0]["id"] == "1"


async def test_lookup_returns_none_after_ttl_expires(db_session: AsyncSession):
    await cache_service.write(db_session, "expired", {"x": 1}, ttl_seconds=-1)
    await db_session.commit()
    hit = await cache_service.lookup(db_session, "expired")
    assert hit is None
