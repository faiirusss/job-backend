from datetime import UTC, datetime, timedelta

import pytest

from app.models import JobListing, MatchResult
from app.schemas import NormalizedJob
from app.services import job_detail_service
from app.services.match_service import is_match_current


class _Session:
    def __init__(self):
        self.flush_count = 0

    async def flush(self):
        self.flush_count += 1


def _job(**overrides) -> JobListing:
    values = {
        "external_id": "441",
        "portal": "linkedin",
        "title": "Backend Engineer",
        "company": "Acme",
        "apply_url": "https://www.linkedin.com/jobs/view/441",
        "description": "",
        "detail_json": None,
        "embedding": [0.1] * 384,
        "responsibilities": ["stale"],
        "mandatory_requirements": ["stale"],
        "skills_tags": ["stale"],
        "scraped_at": datetime.now(UTC) - timedelta(days=1),
    }
    values.update(overrides)
    return JobListing(**values)


def test_needs_linkedin_detail_only_for_incomplete_linkedin_rows():
    assert job_detail_service._needs_linkedin_detail(_job()) is True
    assert job_detail_service._needs_linkedin_detail(_job(portal="glints")) is False
    assert (
        job_detail_service._needs_linkedin_detail(
            _job(description="Full JD", detail_json={"id": "441"})
        )
        is False
    )


def test_apply_linkedin_detail_updates_job_and_invalidates_derived_fields():
    job = _job()
    parsed = {
        "description": "Build APIs with Python.",
        "seniority": "senior",
        "detail": NormalizedJob(id="441", title="Backend Engineer", description_html="<p>JD</p>"),
    }

    changed = job_detail_service._apply_linkedin_detail(job, parsed)

    assert changed is True
    assert job.description == "Build APIs with Python."
    assert job.seniority == "senior"
    assert job.detail_json is not None
    assert job.embedding is None
    assert job.responsibilities is None
    assert job.mandatory_requirements is None
    assert job.skills_tags is None
    assert job.scraped_at is not None


@pytest.mark.asyncio
async def test_ensure_job_detail_flushes_when_fetch_changes_job(monkeypatch):
    async def _fetch(job_id: str, *, company: str = ""):
        return {
            "description": f"Detail for {job_id} at {company}",
            "detail": NormalizedJob(id=job_id, title="Backend Engineer"),
        }

    monkeypatch.setattr(job_detail_service, "fetch_linkedin_detail", _fetch)
    session = _Session()
    job = _job(company="Acme")

    changed = await job_detail_service.ensure_job_detail(session, job)  # type: ignore[arg-type]

    assert changed is True
    assert session.flush_count == 1
    assert job.description == "Detail for 441 at Acme"


def test_match_created_before_lazy_detail_refresh_is_stale():
    now = datetime.now(UTC)
    job = _job(scraped_at=now)
    stale = MatchResult(created_at=now - timedelta(seconds=1), match_score=80)
    current = MatchResult(created_at=now + timedelta(seconds=1), match_score=80)

    assert is_match_current(stale, job) is False
    assert is_match_current(current, job) is True
