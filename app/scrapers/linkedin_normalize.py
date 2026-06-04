"""Pure normalization of LinkedIn guest-endpoint HTML into JobListingDTO /
NormalizedJob. No network, no I/O — every function is deterministic.

Sources:
- listing fragment from .../seeMoreJobPostings/search  → parse_listing_cards
- detail fragment from   .../jobPosting/{id}           → extract_linkedin_job
"""

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup
from loguru import logger

from app.schemas import (
    JobListingDTO,
    NJCategory,
    NJCompany,
    NJLocation,
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

# Tags we keep in description HTML; everything else is unwrapped (text kept),
# matching the tag vocabulary the Glints normalizer emits + LinkedIn's <br>.
ALLOWED_TAGS = {
    "p",
    "br",
    "ul",
    "ol",
    "li",
    "strong",
    "b",
    "em",
    "i",
    "u",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

# LinkedIn "Seniority level" criterion → our DTO seniority enum.
LINKEDIN_SENIORITY = {
    "internship": "junior",
    "entry level": "junior",
    "associate": "mid",
    "mid-senior level": "senior",
    "director": "senior",
    "executive": "senior",
}


def with_start(url: str, start: int) -> str:
    """Return ``url`` with its ``start`` pagination query param set to ``start``."""
    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["start"] = str(start)
    return urlunparse(parsed._replace(query=urlencode(qs)))


def sanitize_html(raw: str) -> str:
    """Keep only ALLOWED_TAGS (stripped of all attributes); unwrap any other tag
    (preserving its text); fully remove <script>/<style>."""
    soup = BeautifulSoup(raw or "", "html.parser")
    for bad in soup(["script", "style"]):
        bad.decompose()
    for tag in soup.find_all(True):
        if tag.name not in ALLOWED_TAGS:
            tag.unwrap()
        else:
            tag.attrs = {}
    return str(soup).strip()


def _job_id_from_urn(urn: str) -> str:
    return urn.rsplit(":", 1)[-1] if urn else ""


# Free-text location forms that don't comma-split into city/country cleanly.
_LOC_STATIC: dict[str, tuple[str, str | None, str | None]] = {
    "jakarta metropolitan area": ("Jakarta", "DKI Jakarta", "Indonesia"),
}


def normalize_location(raw: str) -> NJLocation:
    """Best-effort structuring of LinkedIn's free-text location string.
    Known non-comma forms map via _LOC_STATIC; otherwise comma-split into
    city (first part) / country (last part, defaulting to Indonesia for a
    single token). Always preserves the raw text in ``name``."""
    text = (raw or "").strip()
    if not text:
        return NJLocation()
    static = _LOC_STATIC.get(text.lower())
    if static:
        city, province, country = static
        return NJLocation(name=text, city=city, province=province, country=country)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return NJLocation(name=text)
    return NJLocation(
        name=text,
        city=parts[0],
        province=None,
        country=parts[-1] if len(parts) > 1 else "Indonesia",
    )


def _applicants_count(soup: BeautifulSoup) -> int | None:
    """Parse the first integer out of ``span.num-applicants__caption``
    ("56 applicants" → 56, "Over 200 applicants" → 200); None if absent."""
    el = soup.select_one("span.num-applicants__caption")
    if not el:
        return None
    m = re.search(r"\d+", el.get_text(" ", strip=True))
    return int(m.group()) if m else None


def parse_listing_cards(html: str) -> list[JobListingDTO]:
    """Parse a seeMoreJobPostings/search HTML fragment into listing-level DTOs.
    Score/summary/detail fields are left empty; the pipeline fills them later."""
    soup = BeautifulSoup(html or "", "html.parser")
    out: list[JobListingDTO] = []
    for card in soup.select("div[data-entity-urn]"):
        try:
            job_id = _job_id_from_urn(str(card.get("data-entity-urn", "")))
            title_el = card.select_one("h3.base-search-card__title")
            title = title_el.get_text(strip=True) if title_el else ""
            if not job_id or not title:
                continue

            comp_el = card.select_one("h4.base-search-card__subtitle a") or card.select_one(
                "h4.base-search-card__subtitle"
            )
            company = comp_el.get_text(strip=True) if comp_el else "Unknown"

            loc_el = card.select_one(".job-search-card__location")
            location = loc_el.get_text(strip=True) if loc_el else "Indonesia"

            link_el = card.select_one("a.base-card__full-link")
            href = (str(link_el.get("href") or "") if link_el else "") or (
                f"https://www.linkedin.com/jobs/view/{job_id}"
            )
            apply_url = href.split("?")[0]

            time_el = card.select_one("time")
            posted = str(time_el.get("datetime") or "") if time_el else ""

            # Logo is lazy-loaded: real URL lives in data-delayed-url, not src.
            logo_el = card.select_one("img[data-delayed-url]") or card.select_one("img")
            logo_url = (
                str(logo_el.get("data-delayed-url") or logo_el.get("src") or "") or None
                if logo_el
                else None
            )

            out.append(
                JobListingDTO(
                    id=job_id,
                    portal="linkedin",
                    title=title,
                    company=company or "Unknown",
                    company_logo_bg=logo_bg(company or "Unknown"),
                    company_logo_url=logo_url,
                    location=location or "Indonesia",
                    # Guest listing HTML has no structured work-arrangement field;
                    # infer from location text (only catches literal Remote/Hybrid).
                    work_type=map_work_type(location),  # type: ignore[arg-type]
                    seniority=infer_seniority(title),  # type: ignore[arg-type]
                    salary_min=0,
                    salary_max=0,
                    posted_date=(posted[:10] or "2026-01-01"),
                    posted_label=(posted[:10] or "recent"),
                    apply_url=apply_url,
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
            logger.warning(f"linkedin: failed to parse card: {e}")
    return out


def _criteria(soup: BeautifulSoup) -> dict[str, str]:
    """Map the job-criteria list to {subheader_lowercased: value}."""
    out: dict[str, str] = {}
    for item in soup.select("li.description__job-criteria-item"):
        head = item.select_one(".description__job-criteria-subheader")
        val = item.select_one(".description__job-criteria-text")
        if head and val:
            out[head.get_text(strip=True).lower()] = val.get_text(strip=True)
    return out


def extract_linkedin_job(html: str, job_id: str = "") -> NormalizedJob | None:
    """Build a NormalizedJob from a jobPosting/{id} HTML fragment. Returns None
    if the description markup (.show-more-less-html__markup) is absent."""
    soup = BeautifulSoup(html or "", "html.parser")
    markup = soup.select_one(".show-more-less-html__markup")
    if markup is None:
        return None

    crit = _criteria(soup)
    title_el = soup.select_one(".top-card-layout__title")
    org_el = soup.select_one("a.topcard__org-name-link")
    logo_el = soup.select_one("img.artdeco-entity-image")

    emp = crit.get("employment type", "")
    emp_key = emp.upper().replace("-", "_").replace(" ", "_") if emp else ""
    func = crit.get("job function", "")
    industries = crit.get("industries", "")
    url = f"https://www.linkedin.com/jobs/view/{job_id}" if job_id else ""

    loc_el = soup.select_one(".topcard__flavor--bullet")
    location = normalize_location(loc_el.get_text(strip=True)) if loc_el else NJLocation()

    company = NJCompany(
        name=(org_el.get_text(strip=True) if org_el else ""),
        website=(str(org_el.get("href")) if org_el and org_el.get("href") else None),
        logo_url=(
            str(logo_el.get("data-delayed-url"))
            if logo_el and logo_el.get("data-delayed-url")
            else None
        ),
    )
    return NormalizedJob(
        id=job_id,
        title=(title_el.get_text(strip=True) if title_el else ""),
        canonical_url=url,
        apply_url=url,
        job_type=emp_key,
        job_type_label=enum_label(emp_key, JOB_TYPE_LABEL) if emp_key else "",
        category=NJCategory(name=func, breadcrumb=[c for c in (func, industries) if c]),
        description_html=sanitize_html(str(markup.decode_contents())),
        requirements_html=None,
        salary=NJSalary(),
        location=location,
        company=company,
        applicants_count=_applicants_count(soup),
    )


def _seniority_from_criteria(crit: dict[str, str]) -> str:
    return LINKEDIN_SENIORITY.get(crit.get("seniority level", "").lower(), "")


def parse_job_detail(html: str, job_id: str = "") -> dict[str, Any]:
    """Mirror of glints._parse_job_detail: return a dict the scraper folds into
    the listing DTO. Returns:
      - detail:      the full NormalizedJob for the modal
      - description: plain-text description (for embedding/matching downstream)
      - seniority:   DTO seniority enum derived from the criteria ("" if unknown)
    Empty dict when the payload isn't a recognizable detail."""
    nj = extract_linkedin_job(html, job_id)
    if nj is None:
        return {}
    soup = BeautifulSoup(html or "", "html.parser")
    markup = soup.select_one(".show-more-less-html__markup")
    description = markup.get_text("\n", strip=True) if markup else ""
    return {
        "detail": nj,
        "description": description,
        "seniority": _seniority_from_criteria(_criteria(soup)),
    }
