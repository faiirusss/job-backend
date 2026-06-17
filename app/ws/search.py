from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.db import SessionLocal
from app.events import bus
from app.services import auth_service
from app.services.search_service import get_owned_search_query

router = APIRouter()


@router.websocket("/ws/search")
async def ws_search(websocket: WebSocket, query_id: str) -> None:
    token = websocket.cookies.get(auth_service.COOKIE_NAME)
    async with SessionLocal() as session:
        try:
            user = await auth_service.get_user_by_session_token(session, token)
        except auth_service.InvalidSessionError:
            await websocket.close(code=1008)
            return
        query = await get_owned_search_query(session, user.id, query_id)
        if query is None:
            await websocket.close(code=1008)
            return
        internal_query_id = query.id

    await websocket.accept()
    subscriber = bus.subscribe(internal_query_id)
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
