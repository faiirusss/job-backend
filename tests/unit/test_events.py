import pytest

from app.events import EventBus


@pytest.mark.asyncio
async def test_publish_subscribe_roundtrip():
    bus = EventBus()
    bus.open(42)
    sub = bus.subscribe(42)
    await bus.publish(42, {"type": "status", "message": "hi"})
    await bus.close(42)
    received = []
    async for ev in sub:
        received.append(ev)
    assert received == [{"type": "status", "message": "hi"}]


@pytest.mark.asyncio
async def test_subscriber_after_close_gets_nothing():
    bus = EventBus()
    bus.open(7)
    await bus.publish(7, {"type": "status", "message": "first"})
    await bus.close(7)
    sub = bus.subscribe(7)
    received = []
    async for ev in sub:
        received.append(ev)
    assert received == []


@pytest.mark.asyncio
async def test_multiple_subscribers_each_get_all_events():
    bus = EventBus()
    bus.open(1)
    s1 = bus.subscribe(1)
    s2 = bus.subscribe(1)
    await bus.publish(1, {"type": "status", "message": "a"})
    await bus.publish(1, {"type": "complete", "total": 0, "durationMs": 1.0})
    await bus.close(1)
    r1 = [ev async for ev in s1]
    r2 = [ev async for ev in s2]
    assert len(r1) == 2 and len(r2) == 2
