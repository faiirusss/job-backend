import asyncio
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger


def _describe(event: dict[str, Any]) -> str:
    """Render a published event as a short, human-scannable one-liner for the
    server terminal. Keeps full payloads (job DTOs, params) out of the logs."""
    et = event.get("type", "event")
    if et == "status":
        return event.get("message", "")
    if et == "intro":
        return "intro generated"
    if et == "params":
        p = event.get("payload") or {}
        roles = ", ".join(p.get("role_keywords") or [])
        return f"roles=[{roles}] location={p.get('location')} work={p.get('work_type')}"
    if et == "portal_start":
        return f"portal '{event.get('portal')}' started"
    if et == "progress":
        return f"{event.get('portal')}: scraped {event.get('scraped')}/{event.get('total')}"
    if et == "portal_complete":
        return f"portal '{event.get('portal')}' complete"
    if et == "partial_result":
        j = event.get("job") or {}
        score = j.get("match_score")
        suffix = f" (score {score})" if score is not None else ""
        return f"{j.get('title')} @ {j.get('company')}{suffix}"
    if et == "match":
        return f"scored {len(event.get('job_ids') or [])} job(s)"
    if et == "complete":
        return f"done — {event.get('total')} job(s) in {event.get('durationMs')}ms"
    if et == "error":
        return f"[{event.get('severity')}] {event.get('message')}"
    return ""


def _log_event(query_id: int, event: dict[str, Any]) -> None:
    et = event.get("type", "event")
    desc = _describe(event)
    msg = f"[q{query_id}] {et}" + (f" — {desc}" if desc else "")
    severity = event.get("severity")
    level = "ERROR" if severity == "error" else "WARNING" if severity == "warning" else "INFO"
    logger.bind(query_id=query_id, event_type=et).log(level, msg)


class _Channel:
    def __init__(self) -> None:
        self.subscribers: list[asyncio.Queue[Any]] = []
        self.closed = False
        self.history: list[dict[str, Any]] = []

    async def publish(self, event: dict[str, Any]) -> None:
        self.history.append(event)
        for q in self.subscribers:
            await q.put(event)

    async def close(self) -> None:
        self.closed = True
        for q in self.subscribers:
            await q.put(None)


class EventBus:
    def __init__(self) -> None:
        self._channels: dict[int, _Channel] = {}

    def open(self, query_id: int) -> None:
        channel = self._channels.get(query_id)
        if channel is None or channel.closed:
            self._channels[query_id] = _Channel()

    async def publish(self, query_id: int, event: dict[str, Any]) -> None:
        # Log every event to stdout before the subscriber check, so live progress
        # is visible on the server terminal even if no WS client is attached or
        # the socket has dropped.
        _log_event(query_id, event)
        ch = self._channels.get(query_id)
        if ch is None or ch.closed:
            return
        await ch.publish(event)

    async def close(self, query_id: int) -> None:
        ch = self._channels.get(query_id)
        if ch is None or ch.closed:
            return
        await ch.close()

    def subscribe(self, query_id: int) -> AsyncIterator[dict[str, Any]]:
        ch = self._channels.get(query_id)
        q: asyncio.Queue[Any] = asyncio.Queue()

        if ch is not None:
            # Replay history so far. If the pipeline already completed, this
            # still lets a slightly late WebSocket receive the full run.
            for past in ch.history:
                q.put_nowait(past)
        if ch is not None and not ch.closed:
            ch.subscribers.append(q)
        # elif ch is None or ch.closed: queue stays empty, gen returns immediately

        async def gen() -> AsyncIterator[dict[str, Any]]:
            if ch is None:
                return
            try:
                while True:
                    # If channel already closed AND queue is empty, stop
                    if ch.closed and q.empty():
                        return
                    item = await q.get()
                    if item is None:
                        return
                    yield item
            finally:
                if ch is not None and q in ch.subscribers:
                    ch.subscribers.remove(q)

        return gen()

    def drop(self, query_id: int) -> None:
        self._channels.pop(query_id, None)


bus = EventBus()
