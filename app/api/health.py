import time

from fastapi import APIRouter

router = APIRouter()
_START = time.monotonic()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0", "uptime_seconds": int(time.monotonic() - _START)}
