import pytest

from app.ai.fallback import CircuitBreaker, FallbackLLM, is_rate_limit_error
from app.ai.llm import CoverLetterPair
from app.schemas import JobListingDTO, SearchParams


class _RateLimitedLLM:
    async def parse_intent(self, query: str) -> SearchParams:
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    async def generate_intro(self, query: str, params: SearchParams) -> str:
        raise RuntimeError("429 rate limit")

    async def score_jobs(self, cv_text: str, jobs: list[JobListingDTO]):
        raise RuntimeError("too many requests")

    async def extract_jd_fields(self, jobs: list[JobListingDTO]):
        raise RuntimeError("tokens per minute exceeded")

    async def generate_cover_letter(
        self, cv_text: str, job: JobListingDTO, matched_skills: list[str]
    ) -> CoverLetterPair:
        raise RuntimeError("requests per day exceeded")


class _HealthyLLM:
    async def parse_intent(self, query: str) -> SearchParams:
        return SearchParams(role_keywords=["backend"], location=["Indonesia"])

    async def generate_intro(self, query: str, params: SearchParams) -> str:
        return "fallback intro"

    async def score_jobs(self, cv_text: str, jobs: list[JobListingDTO]):
        return []

    async def extract_jd_fields(self, jobs: list[JobListingDTO]):
        return []

    async def generate_cover_letter(
        self, cv_text: str, job: JobListingDTO, matched_skills: list[str]
    ) -> CoverLetterPair:
        return CoverLetterPair("id", "en", 1, 1)


class _BrokenLLM:
    async def parse_intent(self, query: str) -> SearchParams:
        raise RuntimeError("invalid json")


class _CountingRateLimited:
    """Rate-limited provider that records how many times it was actually invoked.

    Used to prove the circuit breaker stops *calling* a tripped provider, rather
    than merely catching its error after paying the retry cost every time.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def parse_intent(self, query: str) -> SearchParams:
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")


def test_is_rate_limit_error_detects_quota_messages():
    assert is_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED: quota exceeded"))
    assert is_rate_limit_error(RuntimeError("HTTP 429 too many requests"))
    assert not is_rate_limit_error(RuntimeError("invalid json"))


async def test_fallback_llm_uses_next_provider_on_rate_limit():
    llm = FallbackLLM([("gemini", _RateLimitedLLM()), ("qwen", _HealthyLLM())])

    params = await llm.parse_intent("backend")

    assert params.role_keywords == ["backend"]


async def test_fallback_llm_does_not_hide_non_limit_errors():
    llm = FallbackLLM([("gemini", _BrokenLLM()), ("qwen", _HealthyLLM())])

    with pytest.raises(RuntimeError, match="invalid json"):
        await llm.parse_intent("backend")


def test_circuit_breaker_trips_then_clears_after_cooldown():
    clock = [0.0]
    cb = CircuitBreaker(cooldown_seconds=60.0, time_fn=lambda: clock[0])

    assert not cb.is_tripped("gemini")
    cb.trip("gemini")
    assert cb.is_tripped("gemini")

    clock[0] = 59.9
    assert cb.is_tripped("gemini")
    clock[0] = 60.0
    assert not cb.is_tripped("gemini")  # cooldown elapsed → eligible for re-probe


def test_circuit_breaker_reset_clears_trip():
    cb = CircuitBreaker(cooldown_seconds=60.0, time_fn=lambda: 0.0)

    cb.trip("gemini")
    assert cb.is_tripped("gemini")
    cb.reset("gemini")
    assert not cb.is_tripped("gemini")


async def test_fallback_skips_tripped_provider_on_next_call():
    clock = [0.0]
    breaker = CircuitBreaker(cooldown_seconds=100.0, time_fn=lambda: clock[0])
    bad = _CountingRateLimited()
    llm = FallbackLLM([("gemini", bad), ("qwen", _HealthyLLM())], breaker=breaker)

    await llm.parse_intent("backend")
    assert bad.calls == 1  # probed once, then tripped

    await llm.parse_intent("backend")
    assert bad.calls == 1  # skipped while tripped — no retry storm


async def test_fallback_reprobes_provider_after_cooldown():
    clock = [0.0]
    breaker = CircuitBreaker(cooldown_seconds=100.0, time_fn=lambda: clock[0])
    bad = _CountingRateLimited()
    llm = FallbackLLM([("gemini", bad), ("qwen", _HealthyLLM())], breaker=breaker)

    await llm.parse_intent("backend")
    assert bad.calls == 1

    clock[0] = 150.0  # cooldown elapsed
    await llm.parse_intent("backend")
    assert bad.calls == 2  # re-probed after recovery window


async def test_fallback_best_effort_when_every_provider_is_tripped():
    clock = [0.0]
    breaker = CircuitBreaker(cooldown_seconds=100.0, time_fn=lambda: clock[0])
    bad = _CountingRateLimited()
    llm = FallbackLLM([("gemini", bad)], breaker=breaker)

    with pytest.raises(RuntimeError):
        await llm.parse_intent("backend")
    assert bad.calls == 1

    # Even though the only provider is tripped, a degraded best-effort attempt
    # still runs — cooldown is an optimization, never a hard outage.
    with pytest.raises(RuntimeError):
        await llm.parse_intent("backend")
    assert bad.calls == 2


def test_get_llm_wraps_configured_fallback_chain(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDERS", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "q-key")

    import importlib

    from app import config

    importlib.reload(config)
    from app.ai import llm as llm_mod

    importlib.reload(llm_mod)
    instance = llm_mod.get_llm()

    assert isinstance(instance, FallbackLLM)
    assert [name for name, _ in instance._providers] == ["gemini", "qwen"]


def test_get_llm_shares_one_breaker_across_calls(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDERS", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "q-key")

    import importlib

    from app import config

    importlib.reload(config)
    from app.ai import llm as llm_mod

    importlib.reload(llm_mod)

    first = llm_mod.get_llm()
    second = llm_mod.get_llm()

    assert isinstance(first, FallbackLLM)
    # A trip recorded during one search must persist into the next, so both
    # routers must share the same process-wide breaker instance.
    assert first._breaker is second._breaker
