"""Pure normalization of a Glints job-detail payload into a NormalizedJob.

Reads ONLY ``props.pageProps.initialData.data`` (the ROOT). Never reads or
returns accessToken/ssrToken/session/user/remoteAddress/creator PII
(JOB_EXTRACTION.md §0). No network, no I/O — every function is deterministic.
"""

import html as html_lib
import json
import re
from typing import Any

from app.schemas import (
    NJBenefit,
    NJCategory,
    NJCompany,
    NJLocation,
    NJSalary,
    NJSkill,
    NJSocial,
    NormalizedJob,
)
from app.scrapers.common import (
    COMPANY_SIZE_LABEL,
    EDUCATION_LABEL,
    JOB_TYPE_LABEL,
    WORK_ARRANGEMENT_LABEL,
    enum_label,
    experience_label,
)

REQ_HEADINGS = {
    "job requirements",
    "requirements",
    "job requirement",
    "kualifikasi",
    "persyaratan",
    "syarat",
}

_INLINE_TAGS = {
    "BOLD": ("<strong>", "</strong>"),
    "ITALIC": ("<em>", "</em>"),
    "UNDERLINE": ("<u>", "</u>"),
}

_HEADER_LEVEL = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def _apply_inline_styles(text: str, ranges: list[dict]) -> str:
    if not ranges:
        return html_lib.escape(text)
    n = len(text)
    opens: list[list[str]] = [[] for _ in range(n + 1)]
    closes: list[list[str]] = [[] for _ in range(n + 1)]
    for r in ranges:
        o, length, style = r.get("offset", 0), r.get("length", 0), r.get("style")
        if style in _INLINE_TAGS and length > 0 and 0 <= o < n:
            opens[o].append(_INLINE_TAGS[style][0])
            closes[min(o + length, n)].append(_INLINE_TAGS[style][1])
    out: list[str] = []
    for i, ch in enumerate(text):
        out.extend(reversed(closes[i]))
        out.extend(opens[i])
        out.append(html_lib.escape(ch))
    out.extend(reversed(closes[n]))
    return "".join(out)


def _blocks_to_html(blocks: list[dict]) -> str:
    parts: list[str] = []
    list_buffer: list[str] = []
    list_tag: str | None = None

    def flush() -> None:
        nonlocal list_buffer, list_tag
        if list_buffer:
            parts.append(f"<{list_tag}>{''.join(list_buffer)}</{list_tag}>")
            list_buffer, list_tag = [], None

    for b in blocks:
        t = b.get("type", "unstyled")
        inner = _apply_inline_styles(b.get("text", ""), b.get("inlineStyleRanges", []))
        if t in ("ordered-list-item", "unordered-list-item"):
            tag = "ol" if t == "ordered-list-item" else "ul"
            if list_tag and list_tag != tag:
                flush()
            list_tag = tag
            if not b.get("text", "").strip():
                continue
            list_buffer.append(f"<li>{inner}</li>")
            continue
        flush()
        if not b.get("text", "").strip():
            continue
        if t.startswith("header-"):
            lvl = _HEADER_LEVEL.get(t.split("-", 1)[1], 3)
            parts.append(f"<h{lvl}>{inner}</h{lvl}>")
        elif t == "blockquote":
            parts.append(f"<blockquote>{inner}</blockquote>")
        elif t == "code-block":
            parts.append(f"<pre><code>{inner}</code></pre>")
        else:
            parts.append(f"<p>{inner}</p>")
    flush()
    return "".join(parts)


def _load_blocks(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    blocks = data.get("blocks") if isinstance(data, dict) else None
    if not isinstance(blocks, list):
        return []
    return [b for b in blocks if isinstance(b, dict)]


def parse_draftjs(raw: str | None) -> str:
    return _blocks_to_html(_load_blocks(raw))


def parse_draftjs_split(raw: str | None) -> tuple[str, str | None]:
    """Return (description_html, requirements_html|None), split at the first
    block whose text matches a Requirements heading."""
    blocks = _load_blocks(raw)
    if not blocks:
        return "", None
    split_idx: int | None = None
    for i, b in enumerate(blocks):
        if b.get("text", "").strip().lower() in REQ_HEADINGS:
            split_idx = i
            break
    if split_idx is None:
        return _blocks_to_html(blocks), None
    return _blocks_to_html(blocks[:split_idx]), _blocks_to_html(blocks[split_idx:])


def draftjs_to_text(raw: str | None) -> str:
    return "\n".join(b.get("text", "") for b in _load_blocks(raw) if b.get("text", "").strip())


GLINTS_OSS_BASE = "https://glints-dashboard.oss-ap-southeast-1.aliyuncs.com"

_SALARY_MODE = {"MONTH": "/bulan", "YEAR": "/tahun"}


def _fmt_money(currency: str, value: int) -> str:
    return f"{currency} {value:,.0f}".replace(",", ".")


def build_salary(root: dict[str, Any]) -> dict[str, Any]:
    show = bool(root.get("shouldShowSalary"))
    salaries = root.get("salaries") or []
    if not show or not salaries or not isinstance(salaries[0], dict):
        return {
            "show": False,
            "min": None,
            "max": None,
            "currency": None,
            "mode": None,
            "label": "Gaji tidak ditampilkan",
        }
    s = salaries[0]
    mn, mx = s.get("minAmount"), s.get("maxAmount")
    cur = s.get("CurrencyCode") or "IDR"
    mode_suffix = _SALARY_MODE.get(s.get("salaryMode") or "", "")
    if mn and mx:
        label = f"{_fmt_money(cur, mn)} – {_fmt_money(cur, mx)}{mode_suffix}"
    elif mn:
        label = f"Mulai {_fmt_money(cur, mn)}{mode_suffix}"
    else:
        return {
            "show": False,
            "min": None,
            "max": None,
            "currency": cur,
            "mode": s.get("salaryMode") or None,
            "label": "Gaji tidak ditampilkan",
        }
    return {
        "show": True,
        "min": mn,
        "max": mx,
        "currency": cur,
        "mode": s.get("salaryMode"),
        "label": label,
    }


def build_image_url(filename: str | None, kind: str = "company-logo") -> str | None:
    if not filename:
        return None
    if filename.startswith("http://") or filename.startswith("https://"):
        return filename
    return f"{GLINTS_OSS_BASE}/{kind}/{filename}"


def parse_social_media(raw: str | None) -> list[dict[str, str]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    out: list[dict[str, str]] = []
    for platform, val in data.items():
        if not val:
            continue
        p = str(platform).lower()
        v = str(val)
        if v.startswith("http"):
            url = v
        elif p == "linkedin":
            url = f"https://www.linkedin.com/company/{v}"
        elif p == "instagram":
            url = f"https://instagram.com/{v}"
        elif p == "twitter":
            url = f"https://twitter.com/{v}"
        elif p == "facebook":
            url = f"https://facebook.com/{v}"
        else:
            url = v
        out.append({"platform": p, "url": url})
    return out


def parse_gallery(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        files = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(files, list):
        return []
    urls: list[str] = []
    for f in files:
        u = build_image_url(f, kind="company-photo") if isinstance(f, str) else None
        if u:
            urls.append(u)
    return urls


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return s or "job"


def build_urls(root: dict[str, Any]) -> tuple[str, str]:
    job_id = root.get("id", "")
    canonical = (
        f"https://glints.com/id/opportunities/jobs/{slugify(root.get('title', ''))}/{job_id}"
    )
    apply_url = root.get("externalApplyURL") or canonical
    return canonical, apply_url


def _location(root: dict[str, Any]) -> NJLocation:
    loc = root.get("location") or {}
    if not isinstance(loc, dict):
        loc = {}
    city = province = country = None
    # Match by administrativeLevelName across location + its parents.
    candidates = [loc] + (loc.get("parents") or [])
    for c in candidates:
        if not isinstance(c, dict):
            continue
        level_name = (c.get("administrativeLevelName") or "").lower()
        name = c.get("name") or c.get("formattedName")
        if level_name == "city" and not city:
            city = name
        elif level_name == "province" and not province:
            province = name
        elif level_name == "country" and not country:
            country = name
    country = country or (
        (root.get("country") or {}).get("name") if isinstance(root.get("country"), dict) else None
    )
    poi = ((root.get("companyAddress") or {}).get("poi") or {}).get("coordinate") or {}
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    return NJLocation(
        name=loc.get("formattedName") or loc.get("name") or "",
        city=city,
        province=province,
        country=country,
        latitude=lat if lat is not None else poi.get("latitude"),
        longitude=lng if lng is not None else poi.get("longitude"),
    )


def _category(root: dict[str, Any]) -> NJCategory:
    cat = root.get("hierarchicalJobCategory") or {}
    if not isinstance(cat, dict):
        return NJCategory()
    parents = [p for p in (cat.get("parents") or []) if isinstance(p, dict)]
    parents.sort(key=lambda p: p.get("level", 0))
    crumb = [p.get("name") or p.get("defaultName") or "" for p in parents]
    name = cat.get("name") or cat.get("defaultName") or ""
    if name:
        crumb = [c for c in crumb if c] + [name]
    return NJCategory(name=name, breadcrumb=crumb)


def _company(root: dict[str, Any]) -> NJCompany:
    co = root.get("company") or {}
    if not isinstance(co, dict):
        co = {}
    industry = co.get("industry")
    return NJCompany(
        name=co.get("displayName") or co.get("name") or "",
        tagline=co.get("tagline"),
        logo_url=build_image_url(co.get("logo"), kind="company-logo"),
        banner_url=build_image_url(co.get("bannerPic"), kind="company-photo"),
        website=co.get("website"),
        industry=industry.get("name") if isinstance(industry, dict) else None,
        size_label=enum_label(co.get("size"), COMPANY_SIZE_LABEL) or None,
        address=co.get("address"),
        description_html=parse_draftjs(co.get("descriptionJsonString")) or None,
        is_verified=co.get("status") == "VERIFIED",
        social_media=[
            NJSocial(**s) for s in parse_social_media(co.get("socialMediaSitesJsonString"))
        ],
        gallery_urls=parse_gallery(co.get("photosJsonString")),
    )


def _skills(root: dict[str, Any]) -> list[NJSkill]:
    out: list[NJSkill] = []
    for s in root.get("skills") or []:
        if not isinstance(s, dict):
            continue
        inner = s.get("skill") or {}
        name = (inner.get("name") if isinstance(inner, dict) else None) or ""
        if name:
            out.append(NJSkill(name=name, must_have=bool(s.get("mustHave"))))
    return out


def _benefits(root: dict[str, Any]) -> list[NJBenefit]:
    out: list[NJBenefit] = []
    for b in root.get("benefits") or []:
        if not isinstance(b, dict):
            continue
        title = b.get("title") or ""
        if title:
            out.append(
                NJBenefit(
                    title=title,
                    description=b.get("description") or "",
                    icon_key=b.get("logo") or "",
                )
            )
    return out


def extract_glints_job(next_data: dict[str, Any]) -> NormalizedJob | None:
    """Build a NormalizedJob from a Glints __NEXT_DATA__ dict. Reads ONLY
    props.pageProps.initialData.data; returns None if that path is absent."""
    root: Any = next_data
    for key in ("props", "pageProps", "initialData", "data"):
        root = root.get(key) if isinstance(root, dict) else None
    if not isinstance(root, dict) or not root.get("id"):
        return None

    desc_html, req_html = parse_draftjs_split(root.get("descriptionJsonString"))
    canonical, apply_url = build_urls(root)
    return NormalizedJob(
        id=str(root.get("id", "")),
        title=root.get("title") or "",
        canonical_url=canonical,
        apply_url=apply_url,
        status=root.get("status") or "",
        job_type=root.get("type") or "",
        job_type_label=enum_label(root.get("type"), JOB_TYPE_LABEL),
        work_arrangement=root.get("workArrangementOption") or "",
        work_arrangement_label=enum_label(
            root.get("workArrangementOption"), WORK_ARRANGEMENT_LABEL
        ),
        is_remote=bool(root.get("isRemote")),
        category=_category(root),
        min_years_experience=root.get("minYearsOfExperience") or 0,
        max_years_experience=root.get("maxYearsOfExperience") or 0,
        experience_label=experience_label(
            root.get("minYearsOfExperience"), root.get("maxYearsOfExperience")
        ),
        education_level=root.get("educationLevel") or "",
        education_level_label=enum_label(root.get("educationLevel"), EDUCATION_LABEL),
        description_html=desc_html,
        requirements_html=req_html,
        skills=_skills(root),
        benefits=_benefits(root),
        salary=NJSalary(**build_salary(root)),
        location=_location(root),
        posted_at=root.get("createdAt"),
        updated_at=root.get("updatedAt"),
        expiry_date=root.get("expiryDate"),
        is_cover_letter_mandatory=bool(root.get("isCoverLetterMandatory")),
        company=_company(root),
    )
