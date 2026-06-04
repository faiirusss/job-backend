"""Pure normalization of LinkedIn Voyager (normalized+json) payloads into
JobListingDTO / NormalizedJob. No network, no I/O — deterministic.

Sources:
- search body from  .../voyager/api/graphql?...voyagerJobsDashJobCards  → parse_voyager_search
- detail body from  .../voyager/api/jobs/jobPostings/{id}              → parse_voyager_detail
"""

import html
import json
import re
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from app.schemas import (
    JobListingDTO,
    NJCategory,
    NJCompany,
    NJSalary,
    NormalizedJob,
)
from app.scrapers.common import (
    JOB_TYPE_LABEL,
    enum_label,
    infer_seniority,
    logo_bg,
    map_work_type,
)
from app.scrapers.linkedin_normalize import LINKEDIN_SENIORITY, normalize_location

_WORKTYPE_RE = re.compile(r"\(\s*(on-?site|remote|hybrid)\s*\)\s*$", re.I)
_WORKTYPE_MAP = {"on-site": "onsite", "onsite": "onsite", "remote": "remote", "hybrid": "hybrid"}


def _as_doc(body: Any) -> dict:
    if isinstance(body, dict):
        return body
    try:
        return json.loads(body) if body else {}
    except (ValueError, TypeError):
        return {}


def _work_type_from_location(text: str) -> str:
    m = _WORKTYPE_RE.search(text or "")
    return _WORKTYPE_MAP.get(m.group(1).lower(), "") if m else ""


def _strip_worktype(text: str) -> str:
    return _WORKTYPE_RE.sub("", text or "").strip()


def _listed_date(footer_items: list) -> str:
    for it in footer_items or []:
        if it.get("type") == "LISTED_DATE" and it.get("timeAt"):
            return datetime.fromtimestamp(it["timeAt"] / 1000, tz=UTC).strftime("%Y-%m-%d")
    return ""


def _company_logo_url(card: dict, included: list) -> str | None:
    try:
        urn = card["logo"]["attributes"][0]["detailData"]["*companyLogo"]
    except (KeyError, IndexError, TypeError):
        return None
    comp = next((e for e in included if e.get("entityUrn") == urn), None)
    if not comp:
        return None
    vi = (comp.get("logoResolutionResult") or {}).get("vectorImage") or {}
    root, arts = vi.get("rootUrl"), vi.get("artifacts") or []
    if not root or not arts:
        return None
    seg = max(arts, key=lambda a: a.get("width", 0)).get("fileIdentifyingUrlPathSegment", "")
    return (root + seg) if seg else None


def _text_to_html(text: str) -> str:
    """Voyager descriptions are plaintext (newline-separated). Render as escaped
    paragraphs so the existing SafeHtml renderer shows them cleanly."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return "".join("<p>" + html.escape(p).replace("\n", "<br>") + "</p>" for p in paras)


def _epoch_ms_to_iso(ms: Any) -> str | None:
    """Epoch-milliseconds int → UTC ISO-8601 string; None for anything else."""
    if not isinstance(ms, int) or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


# Inline Pemberly spans → HTML tag pair. Anything not listed renders as plain text.
_INLINE_TAGS = {"Bold": ("<strong>", "</strong>"), "Italic": ("<em>", "</em>")}


def _attr_kind(a: dict) -> str:
    return ((a.get("type") or {}).get("$type") or "").rsplit(".", 1)[-1]


def _render_attributed(desc: dict) -> str:
    """Render a Voyager/Pemberly attributed-text object into sanitized HTML.

    Recruiters author rich descriptions; the payload carries the formatting out of
    band as ``attributes`` (char-range spans) over a flat ``text``. We map:
    ``ListItem`` → ``<li>`` (grouped into ``<ul>``/``<ol>`` per the enclosing
    ``List.ordered``), ``Bold``/``Italic`` → ``<strong>``/``<em>``, and prose
    outside any list into ``<p>`` blocks (blank line splits paragraphs, a lone
    newline becomes ``<br>``). All text is HTML-escaped; the output stays within the
    tag vocabulary the SafeHtml renderer allows. Falls back to plain paragraphs when
    there are no attributes to act on."""
    text = desc.get("text") or ""
    attrs = desc.get("attributes") or []
    if not text:
        return ""
    if not attrs:
        return _text_to_html(text)

    n = len(text)
    inline: list[tuple[int, int, str, str]] = []
    lists: list[tuple[int, int, bool]] = []
    items: list[tuple[int, int]] = []
    for a in attrs:
        kind = _attr_kind(a)
        start = a.get("start", 0)
        end = start + a.get("length", 0)
        if kind in _INLINE_TAGS:
            o, c = _INLINE_TAGS[kind]
            inline.append((start, end, o, c))
        elif kind == "List":
            ordered = bool((a.get("attributeKindUnion") or {}).get("list", {}).get("ordered"))
            lists.append((start, end, ordered))
        elif kind == "ListItem":
            items.append((start, end))

    def render_inline(lo: int, hi: int) -> str:
        """Escape text[lo:hi] and weave in inline tags clamped to the range."""
        open_at: dict[int, list[str]] = {}
        close_at: dict[int, list[str]] = {}
        for s, e, o, c in inline:
            s2, e2 = max(s, lo), min(e, hi)
            if s2 >= e2:
                continue
            open_at.setdefault(s2, []).append(o)
            close_at.setdefault(e2, []).insert(0, c)  # close in LIFO order
        buf: list[str] = []
        for i in range(lo, hi + 1):
            buf.extend(close_at.get(i, ()))
            if i < hi:
                buf.extend(open_at.get(i, ()))
                buf.append(html.escape(text[i]))
        return "".join(buf)

    def trimmed(lo: int, hi: int) -> tuple[int, int]:
        while lo < hi and text[lo] in " \n\t":
            lo += 1
        while hi > lo and text[hi - 1] in " \n\t":
            hi -= 1
        return lo, hi

    def render_prose(lo: int, hi: int) -> str:
        out: list[str] = []
        start = lo
        for m in re.finditer(r"\n{2,}", text[lo:hi]):
            _emit_para(out, start, lo + m.start())
            start = lo + m.end()
        _emit_para(out, start, hi)
        return "".join(out)

    def _emit_para(out: list[str], lo: int, hi: int) -> None:
        lo, hi = trimmed(lo, hi)
        if lo >= hi:
            return
        out.append("<p>" + render_inline(lo, hi).replace("\n", "<br>") + "</p>")

    parts: list[str] = []
    pos = 0
    for ls, le, ordered in sorted(lists):
        if le <= pos:  # already consumed (e.g. a nested list)
            continue
        if ls > pos:
            parts.append(render_prose(pos, ls))
        tag = "ol" if ordered else "ul"
        lis = [
            "<li>" + render_inline(*trimmed(s, e)) + "</li>"
            for s, e in sorted(items)
            if ls <= s < le and trimmed(s, e)[0] < trimmed(s, e)[1]
        ]
        parts.append(f"<{tag}>" + "".join(lis) + f"</{tag}>")
        pos = le
    if pos < n:
        parts.append(render_prose(pos, n))
    return "".join(parts)


def parse_voyager_detail(body: Any, job_id: str = "") -> dict[str, Any]:
    """Mirror of linkedin_normalize.parse_job_detail for Voyager: return a dict the
    scraper folds into the listing DTO (detail / description / seniority).
    Empty dict when the payload has no job description. Company name is left blank
    here — the scraper fills it from the search card it already has."""
    data = _as_doc(body).get("data") or {}
    desc_obj = data.get("description") or {}
    desc = desc_obj.get("text") or ""
    if not desc:
        return {}
    emp = data.get("formattedEmploymentStatus") or ""
    emp_key = emp.upper().replace("-", "_").replace(" ", "_") if emp else ""
    exp = data.get("formattedExperienceLevel") or ""
    funcs = list(data.get("formattedJobFunctions") or [])
    inds = list(data.get("formattedIndustries") or [])
    url = (
        f"https://www.linkedin.com/jobs/view/{job_id}"
        if job_id
        else (data.get("jobPostingUrl") or "")
    )
    applicants = data.get("estimatedNumberOfApplicants")
    # LinkedIn surfaces the *original* posting date ("1 year ago"); prefer it over a
    # later repost (listedAt) and a bare createdAt.
    posted_at = _epoch_ms_to_iso(
        data.get("originalListedAt") or data.get("listedAt") or data.get("createdAt")
    )
    sal_text = (data.get("formattedSalaryDescription") or "").strip()
    salary = NJSalary(show=True, label=sal_text) if sal_text else NJSalary()
    nj = NormalizedJob(
        id=job_id,
        title=data.get("title") or "",
        canonical_url=url,
        apply_url=url,
        job_type=emp_key,
        job_type_label=(enum_label(emp_key, JOB_TYPE_LABEL) if emp_key else "") or emp,
        category=NJCategory(
            name=(funcs[0] if funcs else ""),
            breadcrumb=[c for c in (*funcs, *inds) if c],
        ),
        description_html=_render_attributed(desc_obj),
        requirements_html=None,
        salary=salary,
        location=normalize_location(data.get("formattedLocation") or ""),
        company=NJCompany(name=""),
        applicants_count=applicants if isinstance(applicants, int) else None,
        posted_at=posted_at,
    )
    return {
        "detail": nj,
        "description": desc,
        "seniority": LINKEDIN_SENIORITY.get(exp.lower(), ""),
    }


def parse_voyager_search(body: Any) -> list[JobListingDTO]:
    """Parse a Voyager job-search response into listing-level DTOs. Score/summary/
    detail fields are left empty; the pipeline fills them later."""
    included = _as_doc(body).get("included", [])
    out: list[JobListingDTO] = []
    for card in (e for e in included if e.get("$type", "").endswith("jobs.JobPostingCard")):
        try:
            urn = card.get("preDashNormalizedJobPostingUrn") or ""
            job_id = urn.rsplit(":", 1)[-1] if urn else ""
            title = card.get("jobPostingTitle") or (card.get("title") or {}).get("text") or ""
            if not job_id or not title:
                continue
            company = (card.get("primaryDescription") or {}).get("text") or "Unknown"
            loc_raw = (card.get("secondaryDescription") or {}).get("text") or ""
            work = _work_type_from_location(loc_raw) or map_work_type(loc_raw)
            posted = _listed_date(card.get("footerItems"))
            out.append(
                JobListingDTO(
                    id=job_id,
                    portal="linkedin",
                    title=title,
                    company=company or "Unknown",
                    company_logo_bg=logo_bg(company or "Unknown"),
                    company_logo_url=_company_logo_url(card, included),
                    location=_strip_worktype(loc_raw) or "Indonesia",
                    work_type=work,  # type: ignore[arg-type]
                    seniority=infer_seniority(title),  # type: ignore[arg-type]
                    salary_min=0,
                    salary_max=0,
                    posted_date=(posted or "2026-01-01"),
                    posted_label=(posted or "recent"),
                    apply_url=f"https://www.linkedin.com/jobs/view/{job_id}",
                    match_score=None,
                    cosine=0.0,
                    llm_score=0,
                    matched_skills=[],
                    missing_skills=[],
                    summary_id="",
                    summary_en="",
                    description="",
                    requirements="",
                )
            )
        except Exception as e:  # noqa: BLE001 — one bad card must not kill the run
            logger.warning(f"linkedin_voyager: failed to parse card: {e}")
    return out
