import time
from typing import Any

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import embeddings_service
from app.ai.llm import LLM, get_llm
from app.config import settings
from app.db import session_scope
from app.events import bus
from app.models import JobListing, MatchResult, SearchQuery, SearchResult
from app.schemas import IntroEvent, JobListingDTO, SearchParams, SearchResultsResponse
from app.scrapers import orchestrator
from app.services import cache_service, cv_service
from app.services.relevance_filter import filter_relevant_jobs
from app.services.search_intent import normalize_search_params
from app.utils.hashing import params_hash


def _dedupe_preserve(*lists: list[str]) -> list[str]:
    """Union string lists, case-insensitively deduped, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for x in lst or []:
            key = x.strip().lower()
            if x and key and key not in seen:
                seen.add(key)
                out.append(x)
    return out


async def _apply_jd_extraction(llm: LLM, jobs: list[JobListingDTO]) -> list[JobListingDTO]:
    """Run the batched LLM structured-data cleaner over scraped jobs and fold the
    results back in. Prose fields come from the LLM; skills_tags/benefits are the
    union of Glints' structured arrays with the LLM's reading of the prose. On any
    failure the raw descriptions are kept untouched (robust fallback)."""
    if not jobs:
        return jobs
    try:
        extractions = await llm.extract_jd_fields(jobs)
    except Exception as e:
        logger.warning(f"jd extraction failed: {e}; keeping raw descriptions only")
        return jobs

    out: list[JobListingDTO] = []
    for job, ext in zip(jobs, extractions, strict=False):
        out.append(
            job.model_copy(
                update={
                    "responsibilities": ext.responsibilities,
                    "mandatory_requirements": ext.mandatory_requirements,
                    "nice_to_have_requirements": ext.nice_to_have_requirements,
                    "skills_tags": _dedupe_preserve(job.skills_tags, ext.skills_tags),
                    "benefits": _dedupe_preserve(job.benefits, ext.benefits),
                }
            )
        )
    # If the LLM returned fewer rows than expected, keep the remainder unchanged.
    if len(out) < len(jobs):
        out.extend(jobs[len(out) :])
    return out


async def create_search_row(
    session: AsyncSession,
    query: str,
    *,
    user_id: int,
    conversation_id: int | None = None,
    parsed_params: dict[str, Any] | None = None,
) -> int:
    row = SearchQuery(
        raw_query=query,
        user_id=user_id,
        conversation_id=conversation_id,
        parsed_params=parsed_params,
        status="queued",
    )
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return row.id


async def get_history(session: AsyncSession, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    stmt = (
        select(SearchQuery)
        .where(SearchQuery.user_id == user_id)
        .order_by(SearchQuery.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "query": r.raw_query,
            "date": r.created_at.isoformat(),
            "count": r.result_count or 0,
            "duration_ms": r.duration_ms or 0,
            "from_cache": bool(r.from_cache),
            "conversation_id": r.conversation_id,
            "status": r.status,
        }
        for r in rows
    ]


async def get_search(session: AsyncSession, user_id: int, query_id: int) -> dict[str, Any] | None:
    row = (
        await session.execute(
            select(SearchQuery).where(SearchQuery.id == query_id, SearchQuery.user_id == user_id)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "id": row.id,
        "raw_query": row.raw_query,
        "parsed_params": row.parsed_params,
        "conversation_id": row.conversation_id,
        "result_count": row.result_count,
        "from_cache": bool(row.from_cache),
        "duration_ms": row.duration_ms,
        "status": row.status,
        "error_message": row.error_message,
    }


def _job_dto_from_row(j: JobListing) -> JobListingDTO:
    from app.schemas import NormalizedJob
    from app.scrapers.common import infer_seniority as _infer_seniority, logo_bg as _logo_bg

    return JobListingDTO(
        id=str(j.id),
        portal=j.portal,  # type: ignore[arg-type]
        title=j.title,
        company=j.company,
        company_logo_bg=_logo_bg(j.company),
        location=j.location or "Indonesia",
        work_type=(j.work_type or "onsite"),  # type: ignore[arg-type]
        seniority=(j.seniority or _infer_seniority(j.title)),  # type: ignore[arg-type]
        salary_min=j.salary_min or 0,
        salary_max=j.salary_max or 0,
        posted_date=j.posted_date.isoformat() if j.posted_date else "2026-01-01",
        posted_label="recent",
        apply_url=j.apply_url,
        match_score=None,
        cosine=0.0,
        llm_score=0,
        matched_skills=[],
        missing_skills=[],
        summary_id="",
        summary_en="",
        description=j.description or "",
        requirements=j.requirements or "",
        responsibilities=j.responsibilities or [],
        mandatory_requirements=j.mandatory_requirements or [],
        nice_to_have_requirements=j.nice_to_have_requirements or [],
        skills_tags=j.skills_tags or [],
        benefits=j.benefits or [],
        detail=(NormalizedJob.model_validate(j.detail_json) if j.detail_json else None),
    )


async def _upsert_job(
    session: AsyncSession, j: JobListingDTO, embedding: list[float] | None
) -> int:
    """Upsert by (portal, external_id), return DB id."""
    from datetime import date

    pd: date | None
    try:
        pd = date.fromisoformat(j.posted_date)
    except Exception:
        pd = None

    detail_json = j.detail.model_dump(mode="json") if j.detail else None

    stmt = (
        insert(JobListing)
        .values(
            external_id=j.id,
            portal=j.portal,
            title=j.title,
            company=j.company,
            location=j.location,
            work_type=j.work_type,
            seniority=j.seniority,
            salary_min=j.salary_min,
            salary_max=j.salary_max,
            salary_currency="IDR",
            description=j.description,
            requirements=j.requirements,
            responsibilities=j.responsibilities or None,
            mandatory_requirements=j.mandatory_requirements or None,
            nice_to_have_requirements=j.nice_to_have_requirements or None,
            skills_tags=j.skills_tags or None,
            benefits=j.benefits or None,
            detail_json=detail_json,
            apply_url=j.apply_url,
            posted_date=pd,
            embedding=embedding,
        )
        .on_conflict_do_update(
            index_elements=["portal", "external_id"],
            set_={
                "title": j.title,
                "company": j.company,
                "description": j.description,
                "requirements": j.requirements,
                "responsibilities": j.responsibilities or None,
                "mandatory_requirements": j.mandatory_requirements or None,
                "nice_to_have_requirements": j.nice_to_have_requirements or None,
                "skills_tags": j.skills_tags or None,
                "benefits": j.benefits or None,
                "detail_json": detail_json,
                "embedding": embedding,
            },
        )
        .returning(JobListing.id)
    )
    res = await session.execute(stmt)
    return res.scalar_one()


async def run_pipeline(
    query_id: int,
    query: str,
    force_refresh: bool,
    *,
    params_override: SearchParams | None = None,
) -> None:
    start = time.monotonic()
    llm: LLM = get_llm()
    try:
        async with session_scope() as session:
            row = (
                await session.execute(select(SearchQuery).where(SearchQuery.id == query_id))
            ).scalar_one_or_none()
            if row is None:
                return
            user_id = row.user_id
            if user_id is None:
                raise RuntimeError("search query has no user owner")
            await session.execute(
                SearchQuery.__table__.update()  # type: ignore[attr-defined]
                .where(SearchQuery.id == query_id)
                .values(status="running", error_message=None)
            )

        await bus.publish(
            query_id, {"type": "status", "message": "Mengekstrak parameter pencarian…"}
        )

        params = normalize_search_params(params_override or await llm.parse_intent(query))
        async with session_scope() as session:
            active_cv = await cv_service.get_active_cv_full(session, user_id)
        if active_cv is None:
            duration_ms = int((time.monotonic() - start) * 1000)
            async with session_scope() as session:
                await session.execute(
                    SearchQuery.__table__.update()  # type: ignore[attr-defined]
                    .where(SearchQuery.id == query_id)
                    .values(status="failed", result_count=0, duration_ms=duration_ms)
                )
            await bus.publish(
                query_id,
                {
                    "type": "error",
                    "severity": "error",
                    "message": "Tidak ada CV. Upload CV terlebih dahulu.",
                },
            )
            await bus.publish(query_id, {"type": "complete", "total": 0, "durationMs": duration_ms})
            return
        async with session_scope() as session:
            await session.execute(
                SearchQuery.__table__.update()  # type: ignore[attr-defined]
                .where(SearchQuery.id == query_id)
                .values(parsed_params=params.model_dump(), cv_id=active_cv.id)
            )

        intro_msg = await llm.generate_intro(query, params)
        await bus.publish(query_id, IntroEvent(message=intro_msg).model_dump())

        ph = params_hash(params.model_dump())

        if not force_refresh:
            async with session_scope() as session:
                cached = await cache_service.lookup(session, ph)
            if cached is not None:
                await _replay_cached(query_id, params, cached, start, cv_id=active_cv.id)
                return

        await bus.publish(query_id, {"type": "params", "payload": params.model_dump()})
        await bus.publish(query_id, {"type": "status", "message": "Membuka portal…"})

        async def on_event(ev: dict[str, Any]) -> None:
            # Scrapers only emit progress/portal_start/portal_complete/error now.
            await bus.publish(query_id, ev)

        scraped = await orchestrator.run_portals(["glints", "linkedin"], params, on_event)
        scraped, filter_stats = filter_relevant_jobs(scraped, params)
        if filter_stats.dropped_total:
            logger.info(
                "filtered irrelevant jobs",
                query_id=query_id,
                dropped=filter_stats.dropped_total,
                kept=filter_stats.output_count,
                role=filter_stats.dropped_role,
                location=filter_stats.dropped_location,
                work_type=filter_stats.dropped_work_type,
            )
            await bus.publish(
                query_id,
                {
                    "type": "status",
                    "message": (
                        f"Menyaring {filter_stats.dropped_total} lowongan yang tidak sesuai "
                        "role/lokasi..."
                    ),
                },
            )

        if not scraped:
            duration_ms = int((time.monotonic() - start) * 1000)
            async with session_scope() as session:
                await session.execute(
                    SearchQuery.__table__.update()  # type: ignore[attr-defined]
                    .where(SearchQuery.id == query_id)
                    .values(
                        from_cache=False,
                        result_count=0,
                        duration_ms=duration_ms,
                        status="completed",
                    )
                )
                await _replace_search_results(session, query_id, [])
            await bus.publish(query_id, {"type": "complete", "total": 0, "durationMs": duration_ms})
            return

        await bus.publish(query_id, {"type": "status", "message": "Menyiapkan daftar lowongan..."})

        job_embs = await embeddings_service.encode([j.description or j.title for j in scraped])
        db_ids: list[int] = []
        async with session_scope() as session:
            for j, emb in zip(scraped, job_embs, strict=True):
                db_id = await _upsert_job(session, j, emb)
                db_ids.append(db_id)

        # Emit DB-id-keyed partial_result events so the frontend can open details
        # and trigger match analysis on demand.
        for j_dto, db_id in zip(scraped, db_ids, strict=True):
            pre_score_dto = j_dto.model_copy(update={"id": str(db_id), "match_score": None})
            await bus.publish(
                query_id,
                {"type": "partial_result", "job": pre_score_dto.model_dump()},
            )

        payload_jobs = [
            job.model_copy(update={"id": str(db_id), "match_score": None}).model_dump()
            for job, db_id in zip(scraped, db_ids, strict=True)
        ]

        duration_ms = int((time.monotonic() - start) * 1000)
        async with session_scope() as session:
            await session.execute(
                SearchQuery.__table__.update()  # type: ignore[attr-defined]
                .where(SearchQuery.id == query_id)
                .values(
                    from_cache=False,
                    result_count=len(scraped),
                    duration_ms=duration_ms,
                    cv_id=active_cv.id,
                    status="completed",
                )
            )
            await _replace_search_results(session, query_id, db_ids)
            await cache_service.write(
                session,
                ph,
                {"params": params.model_dump(), "jobs": payload_jobs},
                ttl_seconds=settings.cache_ttl_seconds,
            )

        await bus.publish(
            query_id, {"type": "complete", "total": len(scraped), "durationMs": duration_ms}
        )
    except Exception as e:
        logger.exception("pipeline failed", query_id=query_id)
        duration_ms = int((time.monotonic() - start) * 1000)
        async with session_scope() as session:
            await session.execute(
                SearchQuery.__table__.update()  # type: ignore[attr-defined]
                .where(SearchQuery.id == query_id)
                .values(
                    status="failed",
                    error_message=f"{type(e).__name__}: {e}",
                    duration_ms=duration_ms,
                )
            )
        await bus.publish(
            query_id, {"type": "error", "severity": "error", "message": f"{type(e).__name__}: {e}"}
        )
        await bus.publish(
            query_id,
            {"type": "complete", "total": 0, "durationMs": duration_ms},
        )
    finally:
        await bus.close(query_id)
        bus.drop(query_id)


async def _replay_cached(
    query_id: int,
    params: SearchParams,
    cached: dict[str, Any],
    start: float,
    *,
    cv_id: int,
) -> None:
    cached_jobs = [
        JobListingDTO.model_validate(j) for j in (cached.get("jobs") or []) if isinstance(j, dict)
    ]
    filtered_jobs, _filter_stats = filter_relevant_jobs(cached_jobs, params)
    jobs = [j.model_dump() for j in filtered_jobs]
    duration_ms = int((time.monotonic() - start) * 1000)

    async with session_scope() as session:
        await session.execute(
            SearchQuery.__table__.update()  # type: ignore[attr-defined]
            .where(SearchQuery.id == query_id)
            .values(
                from_cache=True,
                result_count=len(jobs),
                duration_ms=duration_ms,
                cv_id=cv_id,
                status="completed",
            )
        )
        db_ids = [_safe_int(j.get("id")) for j in jobs]
        await _replace_search_results(session, query_id, [j for j in db_ids if j is not None])

    await bus.publish(query_id, {"type": "status", "message": "Memuat dari cache…"})
    await bus.publish(query_id, {"type": "params", "payload": params.model_dump()})
    for j in jobs:
        await bus.publish(query_id, {"type": "partial_result", "job": j})
    await bus.publish(query_id, {"type": "complete", "total": len(jobs), "durationMs": duration_ms})


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _replace_search_results(
    session: AsyncSession, query_id: int, job_ids: list[int]
) -> None:
    await session.execute(delete(SearchResult).where(SearchResult.query_id == query_id))
    seen: set[int] = set()
    position = 0
    for job_id in job_ids:
        if job_id in seen:
            continue
        seen.add(job_id)
        session.add(SearchResult(query_id=query_id, job_id=job_id, position=position))
        position += 1


async def get_search_results(
    session: AsyncSession, user_id: int, query_id: int
) -> SearchResultsResponse | None:
    query = (
        await session.execute(
            select(SearchQuery).where(SearchQuery.id == query_id, SearchQuery.user_id == user_id)
        )
    ).scalar_one_or_none()
    if query is None:
        return None

    rows = (
        await session.execute(
            select(JobListing)
            .join(SearchResult, SearchResult.job_id == JobListing.id)
            .where(SearchResult.query_id == query_id)
            .order_by(SearchResult.position)
        )
    ).scalars().all()

    match_by_job: dict[int, MatchResult] = {}
    if rows:
        match_rows = (
            await session.execute(
                select(MatchResult)
                .where(
                    MatchResult.user_id == user_id,
                    MatchResult.job_id.in_([job.id for job in rows]),
                )
                .order_by(MatchResult.created_at.desc(), MatchResult.id.desc())
            )
        ).scalars().all()
        for match in match_rows:
            match_by_job.setdefault(match.job_id, match)

    jobs: list[JobListingDTO] = []
    for job in rows:
        dto = _job_dto_from_row(job)
        match = match_by_job.get(job.id)
        if match is not None:
            dto = dto.model_copy(
                update={
                    "match_score": match.match_score,
                    "cosine": float(match.cosine_score or 0.0),
                    "llm_score": match.llm_score or 0,
                    "matched_skills": list(match.matched_skills or []),
                    "missing_skills": list(match.missing_skills or []),
                    "summary_id": match.summary_id or "",
                    "summary_en": match.summary_en or "",
                }
            )
        jobs.append(dto)

    return SearchResultsResponse(
        query_id=query.id,
        conversation_id=query.conversation_id,
        jobs=jobs,
        status=query.status,
        result_count=query.result_count or len(jobs),
        duration_ms=query.duration_ms,
    )
