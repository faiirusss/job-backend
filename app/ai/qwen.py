"""Qwen LLM provider over any OpenAI-compatible endpoint.

Works against OpenRouter or Alibaba Cloud Model Studio (DashScope
compatible-mode) — the endpoint is chosen by ``base_url``. Behavior parity with
``GeminiLLM`` (``app/ai/gemini.py``): same prompts, same batching, same graceful
degradation. Transport is async ``httpx`` against ``{base}/chat/completions``.

Free tiers are heavily rate-limited (HTTP 429 on OpenRouter; ``insufficient_quota``
on Model Studio), so calls retry with exponential backoff and the JSON parsing is
tolerant of models that wrap output in prose or code fences, or emit a separate
``reasoning_content`` step, instead of honoring ``response_format``.
"""

import asyncio
import json
import re
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ai.llm import ChatResolution, CoverLetterPair, JDExtraction, MatchOutput
from app.ai.prompts import (
    CHAT_RESOLUTION_PROMPT,
    COVER_PROMPT,
    INTENT_PROMPT,
    INTRO_PROMPT,
    JD_EXTRACT_PROMPT,
    MATCH_PROMPT,
)
from app.schemas import JobListingDTO, SearchParams

# Mirror Gemini's JD-extraction batching: full description blocks are large, so
# keep batches small to bound prompt size and avoid cross-job contamination.
_EXTRACT_BATCH = 5
_EXTRACT_DESC_CAP = 6000

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _extract_json(text: str) -> Any:
    """Parse JSON out of an LLM text response.

    Handles three cases, in order: clean JSON, a ```json fenced block, and JSON
    embedded in surrounding prose (recover the first balanced ``[...]`` or
    ``{...}`` span). Raises ``ValueError`` if no JSON can be found.
    """
    stripped = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Recover the first balanced array/object from prose-wrapped output.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = stripped.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"no parseable JSON in model response: {text[:200]!r}")


class _RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


class QwenLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        rpm: int = 15,
        enable_thinking: bool | None = None,
    ) -> None:
        self._model = model
        self._enable_thinking = enable_thinking
        self._limiter = _RateLimiter(rpm)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # OpenRouter courtesy attribution headers (optional, harmless).
                "HTTP-Referer": "https://github.com/faiirusss/job-backend",
                "X-Title": "Lamarin AI",
            },
        )

    async def _complete(self, prompt: str, *, temperature: float, json_mode: bool) -> str:
        await self._limiter.acquire()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if self._enable_thinking is not None:
            # Alibaba Model Studio: disabling thinking strips reasoning_content,
            # cutting completion tokens ~10x. Unknown params are ignored by
            # OpenRouter, so this stays safe when pointed elsewhere.
            payload["enable_thinking"] = self._enable_thinking
        if json_mode:
            # Best-effort: free models may ignore this; parsing stays tolerant.
            payload["response_format"] = {"type": "json_object"}
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"qwen endpoint error: {data['error']}")
        return data["choices"][0]["message"]["content"] or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _call_json(self, prompt: str) -> Any:
        text = await self._complete(prompt, temperature=0.0, json_mode=True)
        return _extract_json(text)

    async def parse_intent(self, query: str) -> SearchParams:
        data = await self._call_json(INTENT_PROMPT + query)
        return SearchParams(
            role_keywords=data.get("role_keywords") or [],
            location=data.get("location") or ["Indonesia"],
            work_type=data.get("work_type") or ["remote", "hybrid", "onsite"],
            seniority=data.get("seniority") or None,
            salary_min_idr=data.get("salary_min_idr"),
        )

    async def resolve_chat_message(
        self,
        message: str,
        previous_params: SearchParams | None,
        recent_messages: list[dict[str, str]],
    ) -> ChatResolution:
        prompt = CHAT_RESOLUTION_PROMPT.format(
            previous_params=json.dumps(
                previous_params.model_dump() if previous_params else None, ensure_ascii=False
            ),
            recent_messages=json.dumps(recent_messages[-10:], ensure_ascii=False),
            message=message.replace("{", "{{").replace("}", "}}"),
        )
        data = await self._call_json(prompt)
        raw_params = data.get("params")
        params = SearchParams.model_validate(raw_params) if isinstance(raw_params, dict) else None
        action = data.get("action")
        if action not in {"new_search", "refine_search", "general_chat"}:
            action = "general_chat"
        return ChatResolution(
            action=action,
            params=params,
            response_text=str(data.get("response_text") or ""),
        )

    async def generate_intro(self, query: str, params: SearchParams) -> str:
        safe_query = query.replace("{", "{{").replace("}", "}}")
        prompt = INTRO_PROMPT.format(
            query=safe_query,
            roles=", ".join(params.role_keywords) or "—",
            location=", ".join(params.location) or "Indonesia",
            work_type=", ".join(params.work_type) or "semua",
        )
        try:
            text = await self._complete(prompt, temperature=0.7, json_mode=False)
            text = text.strip().strip('"').strip("'")
            if text:
                return text
        except Exception as exc:
            logger.warning("generate_intro Qwen call failed, using template fallback: {}", exc)
        role = params.role_keywords[0] if params.role_keywords else "pekerjaan"
        loc = ", ".join(params.location) if params.location else "Indonesia"
        return f"Baik, saya akan mencari lowongan {role} di {loc} via Glints."

    async def score_jobs(self, cv_text: str, jobs: list[JobListingDTO]) -> list[MatchOutput]:
        if not jobs:
            return []
        jobs_payload = [
            {
                "id": j.id,
                "title": j.title,
                "company": j.company,
                "description": (j.description or "")[:600],
                "responsibilities": j.responsibilities[:8] if j.responsibilities else [],
                "mandatory_requirements": j.mandatory_requirements[:8]
                if j.mandatory_requirements
                else [],
                "nice_to_have_requirements": j.nice_to_have_requirements[:5]
                if j.nice_to_have_requirements
                else [],
                "skills_tags": j.skills_tags[:15] if j.skills_tags else [],
                "benefits": j.benefits[:5] if j.benefits else [],
            }
            for j in jobs
        ]
        prompt = MATCH_PROMPT.format(cv=cv_text[:2000], jobs=json.dumps(jobs_payload))
        data = await self._call_json(prompt)
        out: list[MatchOutput] = []
        for item in data:
            out.append(
                MatchOutput(
                    llm_score=int(item.get("llm_score", 50)),
                    matched_skills=list(item.get("matched_skills") or []),
                    missing_skills=list(item.get("missing_skills") or []),
                    summary_id=str(item.get("summary_id") or ""),
                    summary_en=str(item.get("summary_en") or ""),
                )
            )
        return out

    async def extract_jd_fields(self, jobs: list[JobListingDTO]) -> list[JDExtraction]:
        """Distil each job's free-form description into structured fields, batched.
        A failed batch degrades to empty extractions for that batch so the
        pipeline always proceeds (the raw ``description`` is preserved)."""
        if not jobs:
            return []

        results: list[JDExtraction] = []
        for start in range(0, len(jobs), _EXTRACT_BATCH):
            batch = jobs[start : start + _EXTRACT_BATCH]
            payload = [
                {
                    "index": i,
                    "title": j.title,
                    "description": (j.description or "")[:_EXTRACT_DESC_CAP],
                }
                for i, j in enumerate(batch)
            ]
            prompt = JD_EXTRACT_PROMPT.format(jobs=json.dumps(payload, ensure_ascii=False))
            try:
                data = await self._call_json(prompt)
                if not isinstance(data, list):
                    raise ValueError(f"expected JSON array, got {type(data).__name__}")
            except Exception as e:
                logger.warning(
                    f"qwen: extract_jd_fields batch [{start}:{start + len(batch)}] "
                    f"failed: {e}; falling back to empty extractions for this batch"
                )
                data = []
            for offset in range(len(batch)):
                item = data[offset] if offset < len(data) and isinstance(data[offset], dict) else {}
                results.append(
                    JDExtraction(
                        responsibilities=[str(x) for x in (item.get("responsibilities") or [])],
                        mandatory_requirements=[
                            str(x) for x in (item.get("mandatory_requirements") or [])
                        ],
                        nice_to_have_requirements=[
                            str(x) for x in (item.get("nice_to_have_requirements") or [])
                        ],
                        skills_tags=[str(x) for x in (item.get("skills_tags") or [])],
                        benefits=[str(x) for x in (item.get("benefits") or [])],
                    )
                )
        return results

    async def generate_cover_letter(
        self, cv_text: str, job: JobListingDTO, matched_skills: list[str]
    ) -> CoverLetterPair:
        prompt = COVER_PROMPT.format(
            cv=cv_text[:2500],
            title=job.title,
            company=job.company,
            description=(job.description or "")[:2000],
            skills=", ".join(matched_skills) if matched_skills else "the candidate's strengths",
        )
        data = await self._call_json(prompt)
        id_text = str(data.get("content_id") or "")
        en_text = str(data.get("content_en") or "")
        return CoverLetterPair(
            content_id=id_text,
            content_en=en_text,
            word_count_id=len(id_text.split()),
            word_count_en=len(en_text.split()),
        )
