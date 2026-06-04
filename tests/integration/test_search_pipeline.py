import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import embeddings_service
from app.events import bus
from app.schemas import JobListingDTO
from app.services import cv_service, search_service

PDF = Path(__file__).parent.parent / "fixtures" / "sample_cv.pdf"


def _fake_job(jid: str) -> JobListingDTO:
    return JobListingDTO(
        id=jid,
        portal="glints",
        title="Backend Engineer",
        company=f"Acme{jid}",
        company_logo_bg="#000",
        location="Jakarta",
        work_type="remote",
        seniority="mid",
        salary_min=10000000,
        salary_max=20000000,
        posted_date="2026-05-25",
        posted_label="2 days ago",
        apply_url="https://example.com",
        match_score=None,
        cosine=0.0,
        llm_score=0,
        matched_skills=[],
        missing_skills=[],
        summary_id="",
        summary_en="",
        description="Python FastAPI PostgreSQL Docker",
        requirements="",
    )


async def _fake_run_portals(portals, params, on_event):
    jobs = [_fake_job("j1"), _fake_job("j2")]
    await on_event({"type": "portal_start", "portal": "glints"})
    for _ in jobs:
        await on_event({"type": "progress", "portal": "glints", "count": 1})
    await on_event({"type": "portal_complete", "portal": "glints"})
    return jobs


async def test_run_pipeline_emits_full_event_sequence(db_session: AsyncSession, monkeypatch):
    embeddings_service.load()
    await cv_service.upload_cv(db_session, "cv.pdf", PDF.read_bytes())
    await db_session.commit()

    from sqlalchemy.ext.asyncio import async_sessionmaker

    import app.db as db_mod

    # Make session_scope() use the test schema engine that db_session is using
    db_mod.engine = db_session.bind  # underlying engine of the test session
    db_mod.SessionLocal = async_sessionmaker(db_session.bind, expire_on_commit=False)

    monkeypatch.setattr("app.services.search_service.orchestrator.run_portals", _fake_run_portals)

    query_id = await search_service.create_search_row(db_session, "Cari backend python remote")
    await db_session.commit()

    bus.open(query_id)
    subscriber = bus.subscribe(query_id)
    pipeline = asyncio.create_task(
        search_service.run_pipeline(query_id, "Cari backend python remote", force_refresh=False)
    )

    received_types: list[str] = []
    async for ev in subscriber:
        received_types.append(ev["type"])
    await pipeline

    assert "status" in received_types
    assert "params" in received_types
    # 2 pre-score + 2 post-score = 4 partial_result events for 2 fake jobs
    assert received_types.count("partial_result") == 4
    assert "match" in received_types
    assert received_types[-1] == "complete"


async def test_run_pipeline_emits_intro_event_after_params(db_session: AsyncSession, monkeypatch):
    """The pipeline must publish exactly one `intro` event after parsing the query."""
    embeddings_service.load()
    await cv_service.upload_cv(db_session, "cv.pdf", PDF.read_bytes())
    await db_session.commit()

    from sqlalchemy.ext.asyncio import async_sessionmaker

    import app.db as db_mod

    db_mod.engine = db_session.bind
    db_mod.SessionLocal = async_sessionmaker(db_session.bind, expire_on_commit=False)

    monkeypatch.setattr("app.services.search_service.orchestrator.run_portals", _fake_run_portals)

    query_id = await search_service.create_search_row(db_session, "Cari React Junior di Jakarta")
    await db_session.commit()

    bus.open(query_id)
    received: list[dict] = []
    subscriber = bus.subscribe(query_id)

    pipeline = asyncio.create_task(
        search_service.run_pipeline(query_id, "Cari React Junior di Jakarta", force_refresh=False)
    )

    async for ev in subscriber:
        received.append(ev)
    await pipeline

    intros = [e for e in received if e.get("type") == "intro"]
    assert len(intros) == 1
    assert "React" in intros[0]["message"] or "Junior" in intros[0]["message"]


async def test_intro_event_unit(monkeypatch):
    """Unit-level check: bus.history contains an intro event after parse_intent,
    exercised without a real database by stubbing out the DB-writing session_scope."""
    import contextlib

    from app.ai.fake_llm import FakeLLM
    from app.events import bus

    query_id = 9001

    # Stub session_scope so the DB update calls are no-ops.
    @contextlib.asynccontextmanager
    async def _fake_session_scope():
        class _FakeSession:
            async def execute(self, *a, **kw):
                return None

            async def add(self, *a, **kw):
                pass

        yield _FakeSession()

    monkeypatch.setattr("app.services.search_service.session_scope", _fake_session_scope)

    # Stub run_portals so the pipeline terminates quickly.
    async def _noop_portals(portals, params, on_event):
        return []

    monkeypatch.setattr("app.services.search_service.orchestrator.run_portals", _noop_portals)

    # Ensure FakeLLM is used.
    monkeypatch.setattr("app.services.search_service.get_llm", lambda: FakeLLM())

    bus.open(query_id)
    collected: list[dict] = []

    sub = bus.subscribe(query_id)

    task = asyncio.create_task(
        search_service.run_pipeline(query_id, "Cari React Junior di Jakarta", False)
    )

    async for ev in sub:
        collected.append(ev)
    await task

    intros = [e for e in collected if e.get("type") == "intro"]
    assert len(intros) == 1, f"Expected 1 intro event, got {len(intros)}: {collected}"
    assert (
        "React" in intros[0]["message"]
        or "Junior" in intros[0]["message"]
        or "Jakarta" in intros[0]["message"]
    )


async def test_pipeline_emits_scored_partial_result_with_db_id(
    db_session: AsyncSession, monkeypatch
):
    """After scoring, each emitted partial_result must use a DB integer id
    (as a string) and carry a non-null match_score on the post-score emission."""
    embeddings_service.load()
    await cv_service.upload_cv(db_session, "cv.pdf", PDF.read_bytes())
    await db_session.commit()

    from sqlalchemy.ext.asyncio import async_sessionmaker

    import app.db as db_mod

    db_mod.engine = db_session.bind
    db_mod.SessionLocal = async_sessionmaker(db_session.bind, expire_on_commit=False)

    async def fake_run_portals(portals, params, on_event):
        return [
            JobListingDTO(
                id="g-fake-1",
                portal="glints",
                title="Test Engineer",
                company="Acme",
                company_logo_bg="#000",
                location="Jakarta",
                work_type="remote",
                seniority="mid",
                salary_min=10_000_000,
                salary_max=15_000_000,
                posted_date="2026-05-25",
                posted_label="recent",
                apply_url="https://example.com",
                match_score=None,
                cosine=0.0,
                llm_score=0,
                matched_skills=[],
                missing_skills=[],
                summary_id="",
                summary_en="",
                description="Build things",
                requirements="Be a person",
            ),
        ]

    monkeypatch.setattr("app.services.search_service.orchestrator.run_portals", fake_run_portals)

    query_id = await search_service.create_search_row(db_session, "Test Engineer in Jakarta")
    await db_session.commit()

    bus.open(query_id)
    received: list[dict] = []
    subscriber = bus.subscribe(query_id)

    pipeline = asyncio.create_task(
        search_service.run_pipeline(query_id, "Test Engineer in Jakarta", force_refresh=False)
    )

    async for ev in subscriber:
        received.append(ev)
    await pipeline

    partials = [e for e in received if e.get("type") == "partial_result"]
    # Expect at least one pre-score and one post-score emission per scraped job.
    assert len(partials) >= 2, f"Expected >= 2 partial_result events, got {len(partials)}"
    # Every emission must have a numeric-string id (no external scraper ids leak through).
    for ev in partials:
        assert ev["job"]["id"].isdigit(), f"non-DB id leaked: {ev['job']['id']}"
    # The last partial_result must have a non-null match_score (post-score emission).
    assert partials[-1]["job"]["match_score"] is not None, (
        "last partial_result has null match_score"
    )
    assert partials[-1]["job"]["match_score"] > 0, "last partial_result match_score not > 0"


async def test_structured_fields_round_trip_through_upsert(db_session: AsyncSession):
    """New metadata fields must survive _upsert_job → _job_dto_from_row round-trip."""
    from app.ai.embeddings import embeddings_service
    from app.models import JobListing
    from app.services.search_service import _job_dto_from_row, _upsert_job

    embeddings_service.load()

    job = JobListingDTO(
        id="detail-round-trip-1",
        portal="glints",
        title="SRE",
        company="TestCo",
        company_logo_bg="#000",
        location="Remote",
        work_type="remote",
        seniority="mid",
        salary_min=0,
        salary_max=0,
        posted_date="2026-01-01",
        posted_label="now",
        apply_url="https://example.com",
        match_score=None,
        cosine=0.0,
        llm_score=0,
        matched_skills=[],
        missing_skills=[],
        summary_id="",
        summary_en="",
        description="Build and maintain infra.",
        requirements="",
        responsibilities=["Monitor systems", "Oncall rotation"],
        mandatory_requirements=["Kubernetes", "Linux"],
        nice_to_have_requirements=["Terraform"],
        skills_tags=["Kubernetes", "Linux", "Python"],
        benefits=["Health Insurance"],
    )
    db_id = await _upsert_job(db_session, job, embedding=None)
    await db_session.commit()

    row = await db_session.get(JobListing, db_id)
    dto = _job_dto_from_row(row)

    assert dto.responsibilities == ["Monitor systems", "Oncall rotation"]
    assert dto.mandatory_requirements == ["Kubernetes", "Linux"]
    assert dto.nice_to_have_requirements == ["Terraform"]
    assert dto.skills_tags == ["Kubernetes", "Linux", "Python"]
    assert dto.benefits == ["Health Insurance"]
