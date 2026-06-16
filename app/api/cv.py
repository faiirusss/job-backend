import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import UserAccount
from app.schemas import CVData
from app.services import cv_service
from app.services.cv_service import CVTooLargeError
from app.utils.pdf import PDFEmptyError

router = APIRouter(prefix="/cv", tags=["cv"])


@router.post("/upload", response_model=CVData, status_code=201)
async def upload_cv(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> CVData:
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_CV", "message": "Only PDF files are accepted"}},
        )
    body = await file.read()
    try:
        return await cv_service.upload_cv(session, current_user.id, file.filename or "cv.pdf", body)
    except CVTooLargeError as e:
        raise HTTPException(
            status_code=400, detail={"error": {"code": "INVALID_CV", "message": str(e)}}
        ) from e
    except PDFEmptyError as e:
        raise HTTPException(
            status_code=422, detail={"error": {"code": "CV_EMPTY", "message": str(e)}}
        ) from e


@router.get("", response_model=CVData | None)
async def get_cv(
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> CVData | None:
    data = await cv_service.get_active_cv(session, current_user.id)
    if data is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NO_CV", "message": "No CV uploaded"}}
        )
    return data


@router.get("/preview")
async def preview(
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> FileResponse:
    full = await cv_service.get_active_cv_full(session, current_user.id)
    if full is None or not os.path.exists(full.file_path):
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NO_CV", "message": "No CV uploaded"}}
        )
    return FileResponse(full.file_path, media_type="application/pdf")


@router.get("/download")
async def download(
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> FileResponse:
    full = await cv_service.get_active_cv_full(session, current_user.id)
    if full is None or not os.path.exists(full.file_path):
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NO_CV", "message": "No CV uploaded"}}
        )
    return FileResponse(full.file_path, media_type="application/pdf", filename=full.filename)


@router.delete("", status_code=204)
async def delete_cv(
    session: AsyncSession = Depends(get_db),
    current_user: UserAccount = Depends(get_current_user),
) -> Response:
    deleted = await cv_service.delete_active_cv(session, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "NO_CV", "message": "No CV uploaded"}}
        )
    return Response(status_code=204)
