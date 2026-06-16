from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import embeddings_service
from app.services import cv_service

PDF = Path(__file__).parent.parent / "fixtures" / "sample_cv.pdf"


async def test_upload_cv_replaces_previous(db_session: AsyncSession, test_user_id: int):
    embeddings_service.load()
    data1 = await cv_service.upload_cv(db_session, test_user_id, "first.pdf", PDF.read_bytes())
    await db_session.commit()
    assert data1.text_length > 0
    assert data1.filename == "first.pdf"

    data2 = await cv_service.upload_cv(db_session, test_user_id, "second.pdf", PDF.read_bytes())
    await db_session.commit()
    assert data2.filename == "second.pdf"

    current = await cv_service.get_active_cv(db_session, test_user_id)
    assert current is not None
    assert current.filename == "second.pdf"


async def test_get_active_returns_none_when_empty(db_session: AsyncSession, test_user_id: int):
    current = await cv_service.get_active_cv(db_session, test_user_id)
    assert current is None
