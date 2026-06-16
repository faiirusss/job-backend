import math
import re
import unicodedata
from dataclasses import dataclass

from app.schemas import JobListingDTO, SearchParams
from app.services.search_intent import ROLE_STOP_TERMS

_WORD_RE = re.compile(r"[a-z0-9.+#-]+")

_NON_CITY_LOCATIONS = {
    "indonesia",
    "remote",
    "remote indonesia",
    "remote-indonesia",
    "remote global",
    "remote-global",
}


@dataclass(frozen=True)
class RelevanceFilterStats:
    input_count: int
    output_count: int
    dropped_role: int = 0
    dropped_location: int = 0
    dropped_work_type: int = 0

    @property
    def dropped_total(self) -> int:
        return self.input_count - self.output_count


def filter_relevant_jobs(
    jobs: list[JobListingDTO], params: SearchParams
) -> tuple[list[JobListingDTO], RelevanceFilterStats]:
    """Drop portal results that clearly do not match the parsed search intent.

    Portal search endpoints are intentionally broad and can return adjacent or
    sponsored jobs. This deterministic filter keeps discovery results aligned
    with explicit user constraints before jobs are embedded, cached, and shown.
    """
    role_groups = _role_groups(params)
    location_terms = _location_terms(params)
    work_types = _work_types(params)

    kept: list[JobListingDTO] = []
    dropped_role = 0
    dropped_location = 0
    dropped_work_type = 0

    for job in jobs:
        if role_groups and not _matches_role(job, role_groups):
            dropped_role += 1
            continue
        if location_terms and not _matches_location(job, location_terms):
            dropped_location += 1
            continue
        if work_types and job.work_type not in work_types:
            dropped_work_type += 1
            continue
        kept.append(job)

    return kept, RelevanceFilterStats(
        input_count=len(jobs),
        output_count=len(kept),
        dropped_role=dropped_role,
        dropped_location=dropped_location,
        dropped_work_type=dropped_work_type,
    )


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(_WORD_RE.findall(ascii_text.lower()))


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(_norm(text))


def _role_groups(params: SearchParams) -> list[list[str]]:
    location_tokens = {t for loc in params.location for t in _tokens(loc)}
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for raw in params.role_keywords:
        tokens = [
            token
            for token in _tokens(raw)
            if token not in ROLE_STOP_TERMS and token not in location_tokens
        ]
        if not tokens:
            continue
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        groups.append(tokens)

    return groups


def _location_terms(params: SearchParams) -> list[list[str]]:
    terms: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in params.location:
        normalized = _norm(raw)
        if not normalized or normalized in _NON_CITY_LOCATIONS:
            continue
        tokens = [
            t
            for t in _tokens(raw)
            if t not in {"area", "city", "dki", "kabupaten", "kota", "metropolitan", "province"}
        ]
        if not tokens or set(tokens) <= {"indonesia"}:
            continue
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        terms.append(tokens)
    return terms


def _work_types(params: SearchParams) -> set[str]:
    values = set(params.work_type or [])
    if not values or values == {"remote", "hybrid", "onsite"}:
        return set()
    return values


def _matches_role(job: JobListingDTO, role_groups: list[list[str]]) -> bool:
    corpus = _job_role_corpus(job)
    corpus_tokens = set(_tokens(corpus))
    matched = 0
    for group in role_groups:
        phrase = " ".join(group)
        if phrase in corpus or all(token in corpus_tokens for token in group):
            matched += 1

    required = 1 if len(role_groups) == 1 else min(len(role_groups), max(2, math.ceil(len(role_groups) * 0.6)))
    return matched >= required


def _matches_location(job: JobListingDTO, location_terms: list[list[str]]) -> bool:
    corpus = _job_location_corpus(job)
    corpus_tokens = set(_tokens(corpus))
    for term in location_terms:
        phrase = " ".join(term)
        if phrase in corpus or all(token in corpus_tokens for token in term):
            return True
    return False


def _job_role_corpus(job: JobListingDTO) -> str:
    detail = job.detail
    detail_bits: list[str] = []
    if detail is not None:
        detail_bits.extend(
            [
                detail.category.name,
                " ".join(detail.category.breadcrumb),
                " ".join(skill.name for skill in detail.skills),
                detail.description_html,
                detail.requirements_html or "",
            ]
        )

    return _norm(
        " ".join(
            [
                job.title,
                job.description,
                job.requirements,
                " ".join(job.responsibilities),
                " ".join(job.mandatory_requirements),
                " ".join(job.nice_to_have_requirements),
                " ".join(job.skills_tags),
                *detail_bits,
            ]
        )
    )


def _job_location_corpus(job: JobListingDTO) -> str:
    bits = [job.location]
    if job.detail is not None:
        loc = job.detail.location
        bits.extend([loc.name, loc.city or "", loc.province or "", loc.country or ""])
    return _norm(" ".join(bits))
