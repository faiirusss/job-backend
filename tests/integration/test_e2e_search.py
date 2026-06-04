"""HTTP + WebSocket API tests for job search (POST /search → /ws/search).

Uses a fake orchestrator so tests do not hit live Glints.

Run:
  cd backend && .venv/bin/pytest tests/integration/test_e2e_search.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ai.embeddings import embeddings_service
from app.schemas import JobListingDTO

PDF = Path(__file__).parent.parent / "fixtures" / "sample_cv.pdf"
SEARCH_QUERY = "backend python remote"


def _fake_job(ext_id: str) -> JobListingDTO:
    return JobListingDTO(
        id=ext_id,
        portal="glints",
        title="Backend Engineer",
        company=f"Acme-{ext_id}",
        company_logo_bg="#000",
        location="Jakarta",
        work_type="remote",
        seniority="mid",
        salary_min=10_000_000,
        salary_max=20_000_000,
        posted_date="2026-05-25",
        posted_label="2 days ago",
        apply_url="https://example.com/apply",
        match_score=None,
        cosine=0.0,
        llm_score=0,
        matched_skills=[],
        missing_skills=[],
        summary_id="",
        summary_en="",
        description="Python FastAPI PostgreSQL",
        requirements="3+ years backend",
    )


async def _fake_run_portals(portals, params, on_event):
    """Scraper contract: progress only; pipeline emits partial_result."""
    jobs = [_fake_job("ext-1"), _fake_job("ext-2")]
    await on_event({"type": "portal_start", "portal": "glints"})
    await on_event({"type": "progress", "portal": "glints", "scraped": 1, "total": 2})
    await on_event({"type": "progress", "portal": "glints", "scraped": 2, "total": 2})
    await on_event({"type": "portal_complete", "portal": "glints"})
    return jobs


@pytest.fixture(autouse=True)
def _reset_event_bus():
    from app.events import bus

    bus._channels.clear()
    yield
    bus._channels.clear()


def _patch_db_for_tests(monkeypatch, db_engine) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    import app.db as db_mod

    db_mod.engine = db_engine
    db_mod.SessionLocal = async_sessionmaker(db_engine, expire_on_commit=False)
    monkeypatch.setattr("app.services.search_service.orchestrator.run_portals", _fake_run_portals)
    embeddings_service.load()


def _make_client(monkeypatch, db_engine) -> TestClient:
    _patch_db_for_tests(monkeypatch, db_engine)
    from app.main import app

    return TestClient(app)


def _upload_cv(client: TestClient) -> None:
    with open(PDF, "rb") as f:
        r = client.post("/api/v1/cv/upload", files={"file": ("cv.pdf", f, "application/pdf")})
    assert r.status_code == 201


def _start_search(client: TestClient, query: str) -> int:
    r = client.post("/api/v1/search", json={"query": query})
    assert r.status_code == 202
    return r.json()["query_id"]


def _consume_search_ws(client: TestClient, query_id: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with client.websocket_connect(f"/ws/search?query_id={query_id}") as ws:
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] == "complete":
                break
    return events


def _jobs_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last partial_result per job id (scored overwrites pre-score)."""
    by_id: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.get("type") == "partial_result":
            by_id[str(ev["job"]["id"])] = ev["job"]
    return list(by_id.values())


def test_search_api_job_query_flow(monkeypatch, db_engine):
    """Full flow in one TestClient session: POST search → WS jobs → REST metadata."""
    with _make_client(monkeypatch, db_engine) as client:
        r = client.post("/api/v1/search", json={"query": SEARCH_QUERY})
        assert r.status_code == 409
        assert r.json()["detail"]["error"]["code"] == "NO_CV"

        _upload_cv(client)

        query_id = _start_search(client, SEARCH_QUERY)
        events = _consume_search_ws(client, query_id)
        types_seen = [ev["type"] for ev in events]
        jobs = _jobs_from_events(events)

        assert "status" in types_seen
        assert "intro" in types_seen
        assert "params" in types_seen
        assert "partial_result" in types_seen
        assert "match" in types_seen
        assert types_seen[-1] == "complete"

        assert len(jobs) == 2
        for job in jobs:
            assert job["portal"] == "glints"
            assert job["match_score"] is not None
            assert 0 <= job["match_score"] <= 100
            assert job["description"]
            assert job["apply_url"].startswith("https://")

        complete = next(ev for ev in events if ev["type"] == "complete")
        assert complete["total"] == 2

        params_ev = next(ev for ev in events if ev["type"] == "params")
        assert params_ev["payload"]["role_keywords"]

        # GET /api/v1/search/{query_id}
        meta = client.get(f"/api/v1/search/{query_id}")
        assert meta.status_code == 200
        meta_body = meta.json()
        assert meta_body["id"] == query_id
        assert meta_body["raw_query"] == SEARCH_QUERY
        assert meta_body["result_count"] == 2

        # GET /api/v1/search/history
        history = client.get("/api/v1/search/history?limit=10")
        assert history.status_code == 200
        rows = history.json()
        assert rows[0]["id"] == query_id
        assert rows[0]["count"] == 2

        # GET /api/v1/jobs/{id}
        job_id = int(jobs[0]["id"])
        detail = client.get(f"/api/v1/jobs/{job_id}")
        assert detail.status_code == 200
        detail_body = detail.json()
        assert int(detail_body["id"]) == job_id
        assert detail_body["match_score"] is not None
