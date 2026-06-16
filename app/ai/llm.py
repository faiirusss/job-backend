from dataclasses import dataclass
from typing import Literal, Protocol

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


@dataclass
class ChatResolution:
    action: Literal["new_search", "refine_search", "general_chat"]
    params: SearchParams | None
    response_text: str


class LLM(Protocol):
    async def parse_intent(self, query: str) -> SearchParams: ...

    async def resolve_chat_message(
        self,
        message: str,
        previous_params: SearchParams | None,
        recent_messages: list[dict[str, str]],
    ) -> ChatResolution: ...

    async def generate_intro(self, query: str, params: SearchParams) -> str: ...

    async def score_jobs(self, cv_text: str, jobs: list[JobListingDTO]) -> list[MatchOutput]: ...

    async def extract_jd_fields(self, jobs: list[JobListingDTO]) -> list["JDExtraction"]:
        """Distil each job's free-form ``description`` block into standardized
        structured fields. Returns one result per job, in input order."""
        ...

    async def generate_cover_letter(
        self, cv_text: str, job: JobListingDTO, matched_skills: list[str]
    ) -> CoverLetterPair: ...


def _build_llm(provider: str) -> LLM:
    from app.config import settings

    if provider == "qwen":
        from app.ai.qwen import QwenLLM

        return QwenLLM(
            api_key=settings.qwen_api_key,
            model=settings.qwen_model,
            base_url=settings.qwen_base_url,
            rpm=settings.qwen_rpm_limit,
            enable_thinking=settings.qwen_enable_thinking,
        )

    if provider == "gemini":
        from app.ai.gemini import GeminiLLM

        return GeminiLLM(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            rpm=settings.gemini_rpm_limit,
        )

    from app.ai.fake_llm import FakeLLM

    return FakeLLM()


_breaker = None


def _get_breaker():
    """Lazily build one process-wide circuit breaker shared by every router.

    Sharing it means a provider tripped during one search stays skipped on the
    next one, instead of each fresh ``get_llm()`` re-probing a known-dead key.
    """
    global _breaker
    if _breaker is None:
        from app.ai.fallback import CircuitBreaker
        from app.config import settings

        _breaker = CircuitBreaker(cooldown_seconds=settings.llm_provider_cooldown_seconds)
    return _breaker


def get_llm() -> LLM:
    from app.config import settings

    provider = settings.resolved_llm_provider
    fallback_providers = settings.resolved_llm_fallback_providers
    primary = _build_llm(provider)

    if fallback_providers:
        from app.ai.fallback import FallbackLLM

        chain = [(provider, primary)]
        chain.extend((name, _build_llm(name)) for name in fallback_providers)
        return FallbackLLM(chain, breaker=_get_breaker())

    return primary
