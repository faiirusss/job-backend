from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.db import SessionLocal
from app.events import bus
from app.models import SearchQuery
from app.services import auth_service

router = APIRouter()


@router.websocket("/ws/search")
async def ws_search(websocket: WebSocket, query_id: int) -> None:
    token = websocket.cookies.get(auth_service.COOKIE_NAME)
    async with SessionLocal() as session:
        try:
            user = await auth_service.get_user_by_session_token(session, token)
        except auth_service.InvalidSessionError:
            await websocket.close(code=1008)
            return
        owned = await session.scalar(
            select(SearchQuery.id).where(SearchQuery.id == query_id, SearchQuery.user_id == user.id)
        )
        if owned is None:
            await websocket.close(code=1008)
            return

    await websocket.accept()
    subscriber = bus.subscribe(query_id)
    try:
        async for event in subscriber:
            await websocket.send_json(event)
            if event.get("type") == "complete":
                break
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
