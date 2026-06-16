"""Unit tests for the Qwen / OpenRouter LLM provider — offline only.

The HTTP transport is not exercised here (that's the optional live smoke test);
these cover provider selection and the JSON-from-model-text parsing helper.
"""

import httpx
import pytest

from app.ai.qwen import QwenLLM, _extract_json


def _qwen_with_transport(handler) -> QwenLLM:
    """A QwenLLM whose HTTP client is backed by a mock transport (no network)."""
    llm = QwenLLM(api_key="k", model="qwen/test")
    llm._client = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k"},
    )
    return llm


def test_get_llm_returns_qwen_for_qwen_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "q-key")
    # config.settings is read at import-time; reload so env takes effect.
    import importlib

    from app import config

    importlib.reload(config)
    from app.ai import llm as llm_mod

    importlib.reload(llm_mod)
    instance = llm_mod.get_llm()
    assert isinstance(instance, QwenLLM)


def test_extract_json_parses_clean_array():
    assert _extract_json('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_strips_code_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_recovers_from_surrounding_prose():
    text = 'Sure! Here is the result:\n[{"llm_score": 80}]\nHope that helps.'
    assert _extract_json(text) == [{"llm_score": 80}]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json("no json at all here")


async def test_parse_intent_unwraps_openrouter_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"role_keywords": ["data analyst"], '
                            '"location": ["Jakarta"], "salary_min_idr": 8000000}'
                        }
                    }
                ]
            },
        )

    llm = _qwen_with_transport(handler)
    params = await llm.parse_intent("data analyst di Jakarta")
    assert params.role_keywords == ["data analyst"]
    assert params.location == ["Jakarta"]
    assert params.salary_min_idr == 8000000
    await llm._client.aclose()


async def test_call_json_raises_on_error_body():
    from tenacity import RetryError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "rate-limited", "code": 429}})

    llm = _qwen_with_transport(handler)
    # An error body is retried (transient provider errors); after exhausting
    # attempts tenacity surfaces a RetryError.
    with pytest.raises(RetryError):
        await llm._call_json("anything")
    await llm._client.aclose()


async def test_extract_jd_fields_degrades_to_empty_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "throttled"})

    from app.schemas import JobListingDTO

    llm = _qwen_with_transport(handler)
    job = JobListingDTO(
        id="1",
        portal="glints",
        title="Analyst",
        company="Acme",
        company_logo_bg="#000000",
        location="Indonesia",
        work_type="remote",
        seniority="mid",
        salary_min=0,
        salary_max=0,
        posted_date="2026-01-01",
        posted_label="recent",
        apply_url="https://glints.com/id/opportunities/jobs/1",
        match_score=None,
        cosine=0.0,
        llm_score=0,
        matched_skills=[],
        missing_skills=[],
        summary_id="",
        summary_en="",
        description="Do analysis. Need SQL.",
        requirements="",
    )
    out = await llm.extract_jd_fields([job])
    # One result per job, empty buckets — raw description is preserved by the pipeline.
    assert len(out) == 1
    assert out[0].responsibilities == []
    assert out[0].skills_tags == []
    await llm._client.aclose()
