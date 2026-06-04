from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.events import bus

router = APIRouter()


@router.websocket("/ws/search")
async def ws_search(websocket: WebSocket, query_id: int) -> None:
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
