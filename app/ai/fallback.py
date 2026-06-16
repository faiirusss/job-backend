import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from tenacity import RetryError

from app.ai.llm import LLM, ChatResolution, CoverLetterPair, JDExtraction, MatchOutput
from app.schemas import JobListingDTO, SearchParams

_RATE_LIMIT_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate-limit",
    "rate_limit",
    "resource_exhausted",
    "too many requests",
    "tokens per minute",
    "requests per day",
    "temporarily rate-limited",
    "insufficient_quota",
)


def _iter_exception_chain(exc: BaseException, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if id(exc) in seen:
        return
    seen.add(id(exc))
    yield exc

    if isinstance(exc, RetryError):
        try:
            inner = exc.last_attempt.exception()
        except Exception:
            inner = None
        if inner is not None:
            yield from _iter_exception_chain(inner, seen)

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        yield from _iter_exception_chain(cause, seen)

    context = getattr(exc, "__context__", None)
    if context is not None:
        yield from _iter_exception_chain(context, seen)


def is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort classifier for provider quota/rate-limit failures.

    Gemini and OpenRouter surface limits through different exception classes.
    We inspect status/code-like attributes, wrapped tenacity errors, and message
    text so the fallback router can switch provider without coupling to either
    SDK's concrete exception hierarchy.
    """
    for current in _iter_exception_chain(exc):
        response = getattr(current, "response", None)
        status_code = (
            getattr(current, "status_code", None)
            or getattr(response, "status_code", None)
            or getattr(current, "code", None)
        )
        if str(status_code) == "429":
            return True

        status = str(getattr(current, "status", "")).lower()
        if status == "resource_exhausted":
            return True

        text = f"{type(current).__name__} {current}".lower()
        if any(marker in text for marker in _RATE_LIMIT_MARKERS):
            return True

    return False


class CircuitBreaker:
    """Process-wide record of which providers are currently rate-limited.

    A provider that raises a quota/rate-limit error is *tripped* for a cooldown
    window and skipped on subsequent calls, so a known-dead provider (e.g. a
    Gemini key whose daily quota is exhausted) is not re-probed — and re-retried
    with backoff — on every LLM call in the pipeline. Once the cooldown elapses
    the provider becomes eligible again and gets one re-probe.
    """

    def __init__(
        self, cooldown_seconds: float = 300.0, time_fn: Callable[[], float] = time.monotonic
    ) -> None:
        self._cooldown = cooldown_seconds
        self._time = time_fn
        self._tripped_until: dict[str, float] = {}

    def is_tripped(self, name: str) -> bool:
        until = self._tripped_until.get(name)
        if until is None:
            return False
        if self._time() >= until:
            del self._tripped_until[name]  # cooldown elapsed → eligible for re-probe
            return False
        return True

    def trip(self, name: str) -> None:
        self._tripped_until[name] = self._time() + self._cooldown

    def reset(self, name: str) -> None:
        self._tripped_until.pop(name, None)


class FallbackLLM:
    """LLM router that tries fallback providers when the active one is rate-limited.

    A shared :class:`CircuitBreaker` remembers which providers are currently
    rate-limited so they are skipped (not re-retried) across calls.
    """

    def __init__(
        self, providers: list[tuple[str, LLM]], breaker: CircuitBreaker | None = None
    ) -> None:
        if not providers:
            raise ValueError("FallbackLLM requires at least one provider")
        self._providers = providers
        self._breaker = breaker if breaker is not None else CircuitBreaker()

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # Skip providers currently in cooldown. If that leaves nothing, fall back
        # to the full chain — cooldown is an optimization, never a hard outage.
        candidates = [p for p in self._providers if not self._breaker.is_tripped(p[0])]
        if not candidates:
            candidates = list(self._providers)

        last_index = len(candidates) - 1
        for idx, (name, provider) in enumerate(candidates):
            try:
                result = await getattr(provider, method)(*args, **kwargs)
                self._breaker.reset(name)
                return result
            except Exception as exc:
                if not is_rate_limit_error(exc):
                    raise
                self._breaker.trip(name)
                if idx == last_index:
                    raise
                next_name = candidates[idx + 1][0]
                logger.warning(
                    "LLM provider {} hit a rate/quota limit during {}; "
                    "tripping it for the cooldown window and falling back to {}",
                    name,
                    method,
                    next_name,
                )
        raise RuntimeError("unreachable fallback provider state")

    async def parse_intent(self, query: str) -> SearchParams:
        return await self._call("parse_intent", query)

    async def resolve_chat_message(
        self,
        message: str,
        previous_params: SearchParams | None,
        recent_messages: list[dict[str, str]],
    ) -> ChatResolution:
        return await self._call("resolve_chat_message", message, previous_params, recent_messages)

    async def generate_intro(self, query: str, params: SearchParams) -> str:
        return await self._call("generate_intro", query, params)

    async def score_jobs(self, cv_text: str, jobs: list[JobListingDTO]) -> list[MatchOutput]:
        return await self._call("score_jobs", cv_text, jobs)

    async def extract_jd_fields(self, jobs: list[JobListingDTO]) -> list[JDExtraction]:
        return await self._call("extract_jd_fields", jobs)

    async def generate_cover_letter(
        self, cv_text: str, job: JobListingDTO, matched_skills: list[str]
    ) -> CoverLetterPair:
        return await self._call("generate_cover_letter", cv_text, job, matched_skills)
