import os
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import embeddings_service
from app.config import settings
from app.models import CV
from app.schemas import CVData
from app.utils.pdf import extract_text

MAX_PDF_BYTES = 5 * 1024 * 1024


class CVTooLargeError(Exception):
    pass


@dataclass
class ActiveCV:
    id: int
    filename: str
    file_path: str
    text_content: str
    embedding: list[float]
    updated_at: datetime


def _format_jakarta(dt: datetime) -> str:
    return dt.strftime("%d %b %Y, %H:%M") + " WIB"


def _to_cv_data(cv: CV) -> CVData:
    return CVData(
        filename=cv.filename,
        size_kb=max(1, os.path.getsize(cv.file_path) // 1024)
        if os.path.exists(cv.file_path)
        else 1,
        updated_at=_format_jakarta(cv.updated_at),
        text_length=len(cv.text_content),
        text_preview=cv.text_content,
    )


async def upload_cv(session: AsyncSession, filename: str, pdf_bytes: bytes) -> CVData:
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise CVTooLargeError(f"CV exceeds {MAX_PDF_BYTES} bytes")

    text = extract_text(pdf_bytes)
    embeddings_service.load()
    [vec] = await embeddings_service.encode([text])

    os.makedirs(settings.cv_files_dir, exist_ok=True)
    safe_name = filename.replace("/", "_")
    file_path = os.path.join(settings.cv_files_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    await session.execute(delete(CV))

    cv = CV(
        filename=safe_name,
        file_path=file_path,
        text_content=text,
        embedding=vec,
    )
    session.add(cv)
    await session.flush()
    await session.refresh(cv)
    return _to_cv_data(cv)


async def get_active_cv(session: AsyncSession) -> CVData | None:
    result = await session.execute(select(CV).order_by(CV.id.desc()).limit(1))
    cv = result.scalar_one_or_none()
    if cv is None:
        return None
    return _to_cv_data(cv)


async def get_active_cv_full(session: AsyncSession) -> ActiveCV | None:
    result = await session.execute(select(CV).order_by(CV.id.desc()).limit(1))
    cv = result.scalar_one_or_none()
    if cv is None:
        return None
    return ActiveCV(
        id=cv.id,
        filename=cv.filename,
        file_path=cv.file_path,
        text_content=cv.text_content,
        embedding=list(cv.embedding),
        updated_at=cv.updated_at,
    )


async def delete_active_cv(session: AsyncSession) -> bool:
    result = await session.execute(select(CV).order_by(CV.id.desc()).limit(1))
    cv = result.scalar_one_or_none()
    if cv is None:
        return False
    file_path = cv.file_path
    await session.execute(delete(CV))
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass
    return True
