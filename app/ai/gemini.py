import asyncio
import json
import re
from typing import Any

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ai.llm import CoverLetterPair, JDExtraction, MatchOutput
from app.schemas import JobListingDTO, SearchParams

_INTENT_PROMPT = """You are a precise intent extractor for an Indonesian job search application.
Extract structured parameters from the user's natural language query.
Output ONLY valid JSON, no preamble.

Schema:
{
  "role_keywords": [string],
  "location": [string],
  "work_type": [string],
  "seniority": [string],
  "salary_min_idr": number | null,
  "salary_max_idr": number | null,
  "language": "id" | "en",
  "follow_up": boolean,
  "confidence": number
}

Query: """

_INTRO_PROMPT = """You are a friendly Indonesian career assistant. The user asked:

"{query}"

You extracted: roles={roles}, location={location}, work_type={work_type}.

Reply with ONE conversational Indonesian sentence (max 25 words) telling them you
are about to search Glints for these jobs. Do not use lists or formatting. Do not
add quotes. Sound natural, like talking to a friend.

Output the sentence only, no preamble."""

_MATCH_PROMPT = """You are an expert career advisor analyzing job-CV fit.
You will be given 1 CV and a batch of jobs with structured fields.
For each job, output JSON with:
- llm_score (int 0-100)
- matched_skills (max 5): skills present in both CV and job (from skills_tags or requirements)
- missing_skills (max 5): critical skills from mandatory_requirements or skills_tags absent in CV
- summary_id (1-2 sentences Bahasa Indonesia): concise gap analysis referencing mandatory requirements
- summary_en (1-2 sentences English): concise gap analysis

Scoring guide:
- Heavily penalize missing mandatory_requirements; lightly penalize missing nice_to_have_requirements
- skills_tags are direct skill signals — match carefully against CV
- responsibilities give day-to-day context; use them to assess culture/role fit

Return a JSON array of objects in input order.

CV:
{cv}

Jobs:
{jobs}
"""

_COVER_PROMPT = """You are a professional career writer. Generate two cover letters for the
candidate (1) Bahasa Indonesia, formal, "Yth. ..." salutation (2) English, professional.
- 250-350 words each
- Mention {company} explicitly
- Highlight these specific skills: {skills}
- Concrete experience, no generic platitudes
- End with call-to-action

Output JSON: {{ "content_id": "...", "content_en": "..." }}

CV:
{cv}

Job:
- Title: {title}
- Company: {company}
- Description: {description}
"""

_JD_EXTRACT_PROMPT = """You are a Structured Data Cleaner for Indonesian job listings.
You receive a batch of job descriptions. Each was authored freely by a recruiter in a
WYSIWYG editor, so headings, formatting, bullet styles, and language (Bahasa Indonesia or
English) vary wildly — many have no section headings at all.

For EACH job, read the full description and intelligently extract these fields:
- responsibilities: day-to-day duties / what the person will actually do
- mandatory_requirements: hard, must-have qualifications and requirements
- nice_to_have_requirements: preferred / bonus / "nilai plus" qualifications
- skills_tags: concrete technical or professional skills named (e.g. "Python", "Figma", "SEO")
- benefits: perks, allowances, insurance, and compensation extras offered

Rules:
- Classify by MEANING, not by heading text. Headings may be missing, misspelled, in
  Indonesian, or merged into prose. Infer the right bucket from the content itself.
- Split run-on or comma-joined blobs into individual items. Strip leading bullets/numbers.
- Keep each item concise (a phrase, not a paragraph) and preserve its original language.
- If a field has no content, return an empty list. NEVER invent items not in the text.

Return a JSON array with exactly one object per job, in the SAME ORDER as the input.

Jobs:
{jobs}
"""


class _JDExtractItem(BaseModel):
    """response_schema for a single job's extracted fields (Gemini structured output)."""

    responsibilities: list[str] = []
    mandatory_requirements: list[str] = []
    nice_to_have_requirements: list[str] = []
    skills_tags: list[str] = []
    benefits: list[str] = []


# Internal batch size for JD extraction. Full description blocks are large, so keep
# batches small to bound prompt size and avoid cross-job contamination.
_EXTRACT_BATCH = 5
# Per-job description cap fed to the extractor; generous enough for full JDs.
_EXTRACT_DESC_CAP = 6000


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


class GeminiLLM:
    def __init__(self, api_key: str, model: str, rpm: int = 15) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._limiter = _RateLimiter(rpm)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _call_json(self, prompt: str, response_schema: Any = None) -> Any:
        await self._limiter.acquire()
        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            # When a schema is supplied Gemini constrains its output to it, which
            # makes the structured-data-cleaner step far more reliable.
            response_schema=response_schema,
        )
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=config,
        )
        text = response.text or ""
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        return json.loads(text)

    async def parse_intent(self, query: str) -> SearchParams:
        data = await self._call_json(_INTENT_PROMPT + query)
        return SearchParams(
            role_keywords=data.get("role_keywords") or [],
            location=data.get("location") or ["Indonesia"],
            work_type=data.get("work_type") or ["remote", "hybrid", "onsite"],
            seniority=data.get("seniority") or None,
            salary_min_idr=data.get("salary_min_idr"),
        )

    async def generate_intro(self, query: str, params: SearchParams) -> str:
        safe_query = query.replace("{", "{{").replace("}", "}}")
        prompt = _INTRO_PROMPT.format(
            query=safe_query,
            roles=", ".join(params.role_keywords) or "—",
            location=", ".join(params.location) or "Indonesia",
            work_type=", ".join(params.work_type) or "semua",
        )
        await self._limiter.acquire()
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,  # warmer for conversational tone
                    response_mime_type="text/plain",
                ),
            )
            text = (response.text or "").strip().strip('"').strip("'")
            if text:
                return text
        except Exception as exc:
            logger.warning("generate_intro Gemini call failed, using template fallback: %s", exc)
        # Fallback if Gemini fails: templated greeting
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
        prompt = _MATCH_PROMPT.format(cv=cv_text[:2000], jobs=json.dumps(jobs_payload))
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
        """Distil each job's free-form description block into structured fields,
        batched and schema-constrained. A failed batch degrades to empty
        extractions for that batch so the pipeline always proceeds (the raw
        ``description`` is preserved regardless)."""
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
            prompt = _JD_EXTRACT_PROMPT.format(jobs=json.dumps(payload, ensure_ascii=False))
            try:
                data = await self._call_json(prompt, response_schema=list[_JDExtractItem])
                if not isinstance(data, list):
                    raise ValueError(f"expected JSON array, got {type(data).__name__}")
            except Exception as e:
                logger.warning(
                    f"gemini: extract_jd_fields batch [{start}:{start + len(batch)}] "
                    f"failed: {e}; falling back to empty extractions for this batch"
                )
                data = []
            # Align by order; pad short responses and ignore extras so the result
            # length always matches the input batch.
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
        prompt = _COVER_PROMPT.format(
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
