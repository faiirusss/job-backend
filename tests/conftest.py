import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Ensure default test env BEFORE app imports. Pin the LLM provider to the offline
# fake regardless of the developer's `.env` (which may select gemini/qwen) — these
# are OS env vars, which take precedence over the `.env` file. Tests that exercise a
# real provider monkeypatch LLM_PROVIDER explicitly.
os.environ.setdefault("USE_FAKE_LLM", "true")
os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("LLM_FALLBACK_PROVIDERS", "")


@pytest.fixture(autouse=True)
def _isolate_storage_state(tmp_path, monkeypatch):
    """Point the LinkedIn/Glints ``*_storage_state_path`` at non-existent tmp
    files so ``storage_state_path()`` deterministically returns ``None`` —
    otherwise, once a developer has seeded a real session
    (``./data/linkedin_state.json``), guest-path tests would unexpectedly enter
    the authenticated (Voyager) branch. Tests that exercise the authenticated
    path monkeypatch ``storage_state_path`` directly, so they are unaffected by
    this default.
    """
    from app.config import settings

    monkeypatch.setattr(
        settings, "linkedin_storage_state_path", str(tmp_path / "no_linkedin_state.json")
    )
    monkeypatch.setattr(
        settings, "glints_storage_state_path", str(tmp_path / "no_glints_state.json")
    )


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator:
    from app.config import settings
    from app.models import Base

    schema = f"test_jhai_{uuid.uuid4().hex[:8]}"
    base_url = settings.database_url

    # First connection: create the schema, then a second connection with the
    # schema in the search_path so CREATE TABLE lands in the right schema.
    bootstrap_engine = create_async_engine(base_url, echo=False)
    async with bootstrap_engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    await bootstrap_engine.dispose()

    # Setup engine: search_path = test schema first, then public (for vector type)
    setup_engine = create_async_engine(
        base_url,
        echo=False,
        connect_args={"server_settings": {"search_path": f"{schema},public"}},
    )
    async with setup_engine.begin() as conn:
        # checkfirst=False: skip existence check so tables are created in the
        # test schema rather than being skipped because public.* already exists
        await conn.run_sync(lambda c: Base.metadata.create_all(c, checkfirst=False))
    await setup_engine.dispose()

    # Test engine: all connections default to the test schema via search_path
    # Include "public" so pgvector types (from the extension) are resolvable
    engine = create_async_engine(
        base_url,
        echo=False,
        connect_args={"server_settings": {"search_path": f"{schema},public"}},
    )

    try:
        yield engine
    finally:
        await engine.dispose()
        cleanup = create_async_engine(base_url, echo=False)
        async with cleanup.begin() as conn:
            await conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        await cleanup.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_user_id(db_session: AsyncSession) -> int:
    from app.models import UserAccount

    user = UserAccount(
        email=f"test-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="test-hash",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user.id
