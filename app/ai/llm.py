from dataclasses import dataclass
from typing import Protocol

from app.schemas import JobListingDTO, SearchParams


@dataclass
class MatchOutput:
    llm_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    summary_id: str
    summary_en: str


@dataclass
class CoverLetterPair:
    content_id: str
    content_en: str
    word_count_id: int
    word_count_en: int


@dataclass
class JDExtraction:
    """Structured fields an LLM distils from a free-form job-description block.

    Every field is a list; an empty list means the description carried no content
    for that bucket. The scraper no longer attempts this split — recruiters author
    descriptions as free WYSIWYG HTML, so an LLM does the cleaning instead.
    """

    responsibilities: list[str]
    mandatory_requirements: list[str]
    nice_to_have_requirements: list[str]
    skills_tags: list[str]
    benefits: list[str]


class LLM(Protocol):
    async def parse_intent(self, query: str) -> SearchParams: ...

    async def generate_intro(self, query: str, params: SearchParams) -> str: ...

    async def score_jobs(self, cv_text: str, jobs: list[JobListingDTO]) -> list[MatchOutput]: ...

    async def extract_jd_fields(self, jobs: list[JobListingDTO]) -> list["JDExtraction"]:
        """Distil each job's free-form ``description`` block into standardized
        structured fields. Returns one result per job, in input order."""
        ...

    async def generate_cover_letter(
        self, cv_text: str, job: JobListingDTO, matched_skills: list[str]
    ) -> CoverLetterPair: ...


def get_llm() -> LLM:
    from app.config import settings

    if settings.use_fake_llm:
        from app.ai.fake_llm import FakeLLM

        return FakeLLM()

    from app.ai.gemini import GeminiLLM

    return GeminiLLM(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        rpm=settings.gemini_rpm_limit,
    )
