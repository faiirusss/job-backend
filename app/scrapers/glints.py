import asyncio
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from loguru import logger

from app.config import settings
from app.schemas import JobListingDTO, SearchParams
from app.scrapers import glints_auth
from app.scrapers.base import BaseScraper, EventEmitter, humanizer_delay
from app.scrapers.common import (
    infer_seniority as _infer_seniority,
    logo_bg as _logo_bg,
    map_work_type as _map_work_type,
    posted_label as _posted_label,
)

_MAX_PAGES = 5
_MAX_JOBS = 100

# Fraction of the per-portal budget reserved for enrichment after listings finish.
# The orchestrator's wait_for cap is a hard backstop; we aim to finish inside it.
_ENRICH_BUDGET_FRACTION = 0.85
# How many detail fetches we dispatch concurrently. In-page fetch() reuses the
# already-authenticated session, so a small fan-out is safe and turns 100 serial
# requests (~100–300 s) into ~5–10 s of wall time.
_ENRICH_CONCURRENCY = 6


def _pick(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-None value from `item` matching any key in `keys`.

    Supports dotted paths like 'company.name' or 'salary.min'.
    """
    for key in keys:
        if "." in key:
            cur: Any = item
            for part in key.split("."):
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    cur = None
                    break
            if cur not in (None, ""):
                return cur
        else:
            v = item.get(key)
            if v not in (None, ""):
                return v
    return default


def _coerce_int(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _parse_html_salary(s: str) -> tuple[int, int]:
    """Parse strings like 'Rp 12 Jt - 18 Jt' or 'IDR 5.000.000 - 8.000.000'.
    Returns (min, max). Either or both may be 0 if not parseable.
    """
    if not s:
        return 0, 0
    # First try to match dot-separated thousands like 5.000.000 or 12.000.000
    # These have the pattern: digits with dots where each segment after first is exactly 3 digits
    dot_thousands = re.findall(r"\b(\d{1,3}(?:\.\d{3})+)\b", s)
    if dot_thousands:
        vals: list[int] = []
        for num in dot_thousands:
            try:
                vals.append(int(num.replace(".", "")))
            except ValueError:
                continue
        if vals:
            if len(vals) == 1:
                return vals[0], 0
            return vals[0], vals[-1]

    # Fall through to Jt/K suffix parsing
    digits = re.findall(r"(\d+(?:[.,]\d+)?)\s*(jt|juta|jt\.|k)?", s.lower())
    vals2: list[int] = []
    for num, unit in digits:
        try:
            base = float(num.replace(",", "."))
        except ValueError:
            continue
        if not base:
            continue
        if unit in ("jt", "juta", "jt."):
            base *= 1_000_000
        elif unit == "k":
            base *= 1_000
        vals2.append(int(base))
    if not vals2:
        return 0, 0
    if len(vals2) == 1:
        return vals2[0], 0
    return vals2[0], vals2[-1]


def _with_page_param(url: str, page: int) -> str:
    """Add or replace a `page` query parameter, preserving everything else."""
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(qs)))


def _graphql_page_body(post_data: str | None, page: int) -> str | None:
    """Given a captured GraphQL search POST body, return a new body string with
    the pagination page set to `page`. Returns None if the captured request is
    not a recognizable GraphQL search body (so callers can fall back to GET).
    """
    if not post_data:
        return None
    try:
        body = json.loads(post_data)
    except (TypeError, ValueError):
        return None
    data = body.get("variables", {}).get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict) or "page" not in data:
        return None
    data["page"] = page
    return json.dumps(body)


def _is_unauthenticated(data: Any) -> bool:
    """True if a Glints reply indicates the session lacks permission (a login-
    gated page), as opposed to a genuine end-of-results empty page. Glints
    signals this via ``isAuthenticated: false`` or an ``errors`` entry with
    ``extensions.code == "NO_PERMISSION"``.
    """
    if not isinstance(data, dict):
        return False
    if data.get("isAuthenticated") is False:
        return True
    errors = data.get("errors")
    if isinstance(errors, list):
        for e in errors:
            if isinstance(e, dict):
                ext = e.get("extensions")
                if isinstance(ext, dict) and ext.get("code") == "NO_PERMISSION":
                    return True
    return False


def _parse_job_detail(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract structured metadata from a Glints job detail __NEXT_DATA__ payload.

    Delegates to glints_normalize.extract_glints_job (reads only
    props.pageProps.initialData.data). Returns a dict with:
      - description:  plain text (draftjs_to_text) for embedding/matching
      - skills_tags:  flat skill names (kept for the matcher's dedup union)
      - benefits:     flat benefit titles (likewise)
      - detail:       the full NormalizedJob for the modal
    Returns {} when the payload isn't a recognizable Glints detail shape.
    """
    from app.scrapers.glints_normalize import draftjs_to_text, extract_glints_job

    nj = extract_glints_job(payload)
    if nj is None:
        return {}

    root = payload["props"]["pageProps"]["initialData"]["data"]
    return {
        "description": draftjs_to_text(root.get("descriptionJsonString")),
        "skills_tags": [s.name for s in nj.skills if s.name],
        "benefits": [b.title for b in nj.benefits if b.title],
        "detail": nj,
    }


def _session_cookies_and_token(path: str) -> tuple[list[dict], str | None]:
    """Read cookies and a best-effort bearer token out of a Playwright
    storage_state file, so a freshly minted session can be applied to a running
    browser context (cookies) and POST-replay headers (token)."""
    try:
        state = json.loads(Path(path).read_text())
    except (ValueError, OSError):
        return [], None
    cookies = state.get("cookies") or []
    token: str | None = None
    for origin in state.get("origins") or []:
        for entry in origin.get("localStorage") or []:
            name = (entry.get("name") or "").lower()
            value = entry.get("value") or ""
            # Glints keeps a JWT in localStorage; JWTs start with "ey" (base64 '{').
            if ("token" in name or "auth" in name) and value.startswith("ey"):
                token = value
                break
        if token:
            break
    return cookies, token


class GlintsScraper(BaseScraper):
    portal = "glints"

    def build_search_url(self, params: SearchParams) -> str:
        keyword = " ".join(params.role_keywords) or "engineer"
        qs = urlencode({"keyword": keyword, "country": "ID"})
        return f"https://glints.com/id/opportunities/jobs/explore?{qs}"

    def _parse_api_response(self, payload: dict[str, Any]) -> list[JobListingDTO]:
        # Glints returns items under multiple shapes depending on the endpoint;
        # try a few known keys.
        items: list[Any] = (
            payload.get("data")
            or payload.get("jobs")
            or payload.get("results")
            or payload.get("items")
            or []
        )
        if isinstance(items, dict):
            # Some shapes wrap the list under `data`. Two cases:
            #   REST:    {"data": {"jobs": [...]}}
            #   GraphQL: {"data": {"searchJobsV3": {"jobsInPage": [...]}}}
            list_keys = ("jobsInPage", "jobs", "results", "items")
            nested: Any = next(
                (items[k] for k in list_keys if isinstance(items.get(k), list)), None
            )
            if nested is None:
                # Descend one level into a GraphQL operation wrapper (e.g. searchJobsV3).
                for v in items.values():
                    if isinstance(v, dict):
                        nested = next(
                            (v[k] for k in list_keys if isinstance(v.get(k), list)),
                            None,
                        )
                        if nested is not None:
                            break
            items = nested or []
        out: list[JobListingDTO] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                ext_id = str(_pick(item, "id", "jobId", default=""))
                title = str(_pick(item, "title", "jobTitle", "name", default=""))

                company_raw = _pick(item, "company", "companyName", "employer", default="")
                if isinstance(company_raw, dict):
                    company = str(_pick(company_raw, "name", "displayName", default="Unknown"))
                else:
                    company = str(company_raw or "Unknown")

                loc_raw = _pick(item, "city", "location", "locationName", default="")
                if isinstance(loc_raw, dict):
                    location = str(
                        _pick(
                            loc_raw,
                            "formattedName",
                            "name",
                            "city",
                            "displayName",
                            default="Indonesia",
                        )
                    )
                else:
                    location = str(loc_raw or "Indonesia")

                desc = str(
                    _pick(
                        item,
                        "description",
                        "jobDescription",
                        "descriptionOriginal",
                        default="",
                    )
                )
                req = str(
                    _pick(
                        item,
                        "requirements",
                        "jobRequirement",
                        "requirementsOriginal",
                        default="",
                    )
                )

                # Salary can be flat (minSalary/maxSalary) or nested (salary.min/max)
                salary_min = _coerce_int(_pick(item, "minSalary", "salary.min", "salaryMin"))
                salary_max = _coerce_int(_pick(item, "maxSalary", "salary.max", "salaryMax"))
                # GraphQL shape: salaries: [{minAmount, maxAmount, ...}]
                if not salary_min and not salary_max:
                    sal_list = _pick(item, "salaries")
                    if isinstance(sal_list, list) and sal_list and isinstance(sal_list[0], dict):
                        salary_min = _coerce_int(sal_list[0].get("minAmount"))
                        salary_max = _coerce_int(sal_list[0].get("maxAmount"))

                apply_url = str(
                    _pick(item, "applyUrl", "url", "jobUrl", default="")
                    or f"https://glints.com/id/opportunities/jobs/{ext_id}"
                )
                created = str(_pick(item, "createdAt", "postedAt", "publishedAt", default=""))
                work_raw = _pick(
                    item, "workArrangementOption", "workArrangement", "workType", default=""
                )

                if not ext_id or not title:
                    logger.warning(
                        f"glints: skipping item missing id/title: id={ext_id!r}, title={title!r}"
                    )
                    continue

                out.append(
                    JobListingDTO(
                        id=ext_id,
                        portal="glints",
                        title=title,
                        company=company or "Unknown",
                        company_logo_bg=_logo_bg(company or "Unknown"),
                        location=location or "Indonesia",
                        work_type=_map_work_type(str(work_raw) if work_raw else None),  # type: ignore[arg-type]
                        seniority=_infer_seniority(title),  # type: ignore[arg-type]
                        salary_min=salary_min,
                        salary_max=salary_max,
                        posted_date=(created[:10] or "2026-01-01"),
                        posted_label=_posted_label(created),
                        apply_url=apply_url,
                        match_score=None,
                        cosine=0.0,
                        llm_score=0,
                        matched_skills=[],
                        missing_skills=[],
                        summary_id="",
                        summary_en="",
                        description=desc,
                        requirements=req,
                    )
                )
            except Exception as e:
                logger.warning(f"glints: failed to parse item: {e}")
        return out

    async def _fetch_job_detail(self, page: Any, job_id: str) -> dict[str, Any]:
        """Fetch the Glints job detail page in-page (inherits session cookies) and
        extract structured data from __NEXT_DATA__ (Next.js initial state)."""
        detail_url = f"https://glints.com/id/opportunities/jobs/{job_id}"
        try:
            html = await page.evaluate(
                "(url) => fetch(url, {credentials: 'include'}).then(r => r.text()).catch(() => '')",
                detail_url,
            )
            if not isinstance(html, str) or not html:
                return {}
            m = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)
            if m:
                try:
                    return json.loads(m.group(1))
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception as e:
            logger.warning(f"glints: detail fetch for {job_id} failed: {e}")
        return {}

    async def _enrich_with_detail(
        self,
        page: Any,
        jobs: list[JobListingDTO],
        *,
        deadline: float | None = None,
        concurrency: int = _ENRICH_CONCURRENCY,
    ) -> list[JobListingDTO]:
        """Enrich job cards with structured detail (description, skills, benefits).

        Fires bounded-concurrency in-page fetches; jobs that don't get enriched
        in time keep their original listing-derived data (no loss).

        ``deadline`` (``loop.time()`` value, optional): stop dispatching new
        detail fetches past this point. The per-job humanizer delay used to live
        here too — it was a 1–3 s sleep between every fetch that made the loop
        unbounded on 100 jobs (the listings already gate themselves on
        pagination, so the in-page detail fetches don't need extra throttling).
        """
        if not jobs:
            return jobs

        loop = asyncio.get_event_loop()
        out: list[JobListingDTO] = list(jobs)  # default → listing data preserved
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _enrich_one(idx: int, job: JobListingDTO) -> None:
            # Cheap pre-check: skip immediately if we're already past the budget
            # so a 100-job run doesn't block on the semaphore queue.
            if deadline is not None and loop.time() >= deadline:
                return
            async with sem:
                if deadline is not None and loop.time() >= deadline:
                    return
                try:
                    raw = await self._fetch_job_detail(page, job.id)
                except Exception as e:
                    logger.warning(f"glints: detail fetch for {job.id} failed: {e}")
                    return
            parsed = _parse_job_detail(raw) if raw else {}
            if parsed:
                # Capture the full detail object + plain-text description + Glints'
                # structured arrays. The LLM extraction step still runs downstream
                # on `description` to feed the matcher's prose fields.
                out[idx] = job.model_copy(
                    update={
                        "description": parsed.get("description") or job.description,
                        "skills_tags": parsed.get("skills_tags", []),
                        "benefits": parsed.get("benefits", []),
                        "detail": parsed.get("detail"),
                    }
                )
            elif raw:
                # The detail page came back but no job object was found at any known
                # path — a signal the Glints payload shape drifted (e.g. moved off
                # __NEXT_DATA__). Surface it so the cause is visible in logs rather
                # than silently leaving the structured fields empty.
                logger.debug(
                    f"glints: detail for {job.id} fetched but _parse_job_detail "
                    f"found no job (payload shape drift?)"
                )

        # return_exceptions=True so one failed fetch never poisons the gather;
        # individual failures are already logged inside _enrich_one.
        await asyncio.gather(
            *(_enrich_one(i, j) for i, j in enumerate(jobs)),
            return_exceptions=True,
        )
        return out

    async def _reauth(self, page: Any, extra_headers: dict[str, str]) -> bool:
        """Re-mint the Glints service-account session and apply it to the live
        context so the in-flight pagination can continue. Cookies go onto the
        browser context (used by ``credentials:'include'``); any bearer token
        goes onto the replayed-POST headers. Returns True if a usable session was
        applied (the caller should retry the page)."""
        new_path = await glints_auth.refresh_session()
        if not new_path:
            return False
        cookies, token = _session_cookies_and_token(new_path)
        applied = False
        if cookies:
            try:
                await page.context.add_cookies(cookies)
                applied = True
            except Exception as e:
                logger.warning(f"glints: add_cookies after refresh failed: {e}")
        if token:
            extra_headers["authorization"] = f"Bearer {token}"
            applied = True
        return applied

    async def scrape(
        self, page: Any, params: SearchParams, on_event: EventEmitter
    ) -> list[JobListingDTO]:
        return await self._scrape_impl(page, params, on_event)

    async def _scrape_impl(
        self,
        page: Any,
        params: SearchParams,
        on_event: EventEmitter,
    ) -> list[JobListingDTO]:
        url = self.build_search_url(params)

        # Cooperative deadline: we aim to finish all listing + enrichment work
        # within ENRICH_BUDGET_FRACTION of the per-portal timeout, leaving the
        # orchestrator's wait_for as a hard backstop. When enrichment runs out
        # of time it returns whatever was already scraped — the 100-jobs-lost
        # bug came from blowing this budget and being externally cancelled.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + settings.scraper_timeout_seconds * _ENRICH_BUDGET_FRACTION

        captured: dict[str, Any] = {}

        async def _on_response(response: Any) -> None:
            try:
                url_l = response.url.lower()
                is_candidate = (
                    "job-listings" in url_l
                    or "/api/v2/" in url_l
                    or "/graphql" in url_l
                    or "/search" in url_l
                    or ("opportunities" in url_l and ("json" in url_l or "api" in url_l))
                )
                if not is_candidate:
                    return
                content_type = (response.headers.get("content-type") or "").lower()
                if "application/json" not in content_type:
                    return
                try:
                    data = await response.json()
                except Exception:
                    return
                if not isinstance(data, dict):
                    return
                # Several Glints GraphQL ops (getJobSearchFilters, searchHierarchical-
                # Locations, getJobTitleSuggestions) also return a top-level "data" key.
                # Only capture a response that actually yields jobs, and never let a
                # later non-job response overwrite the job payload.
                if "payload" in captured:
                    return
                if not self._parse_api_response(data):
                    return
                captured["payload"] = data
                captured["api_url"] = response.url
                # Capture the request so GraphQL POST searches can be replayed
                # with an incremented page for pagination.
                try:
                    captured["post_data"] = response.request.post_data
                except Exception:
                    captured["post_data"] = None
                # Glints requires custom headers (e.g. x-glints-country-code) on the
                # GraphQL POST; forward them when replaying paginated requests.
                try:
                    captured["req_headers"] = await response.request.all_headers()
                except Exception:
                    captured["req_headers"] = None
            except Exception:
                pass

        page.on("response", _on_response)

        await asyncio.sleep(humanizer_delay("page_load"))
        logger.info(f"glints: navigating to {url}")
        # wait_until="commit" resolves as soon as the server response is received,
        # not on DOMContentLoaded. Glints is a Cloudflare-fronted Next.js app whose
        # DOMContentLoaded can take >30 s in headless Chromium (notably over WSL2's
        # IPv6-only route to Glints), but the job data we actually consume arrives
        # via the searchJobsV3 GraphQL XHR within a few seconds. Waiting on the DOM
        # both blocked needlessly and, on timeout, discarded an already-captured
        # payload — the search-returns-nothing bug.
        try:
            await page.goto(url, timeout=30000, wait_until="commit")
        except Exception as e:
            # Navigation may fail (or hang past the timeout) even though the response
            # interceptor already grabbed the job payload. Don't bail here: log,
            # surface a warning, and fall through to the payload poll below. We only
            # give up later if nothing was captured.
            logger.warning(f"glints: navigation incomplete: {e}")
            await on_event(
                {"type": "error", "severity": "warning", "portal": self.portal, "message": str(e)}
            )

        # Condition-based wait for the intercepted XHR instead of a fixed sleep:
        # poll until the job payload lands (or we run out of budget). It typically
        # arrives 2–4 s after commit; capping at the cooperative deadline keeps the
        # orchestrator's wait_for backstop from ever firing first.
        poll_deadline = min(deadline, loop.time() + 20)
        while "payload" not in captured and loop.time() < poll_deadline:
            await asyncio.sleep(0.25)

        all_jobs: list[JobListingDTO] = []
        seen_ext_ids: set[str] = set()

        # Page 1 — from the intercepted XHR if we got it
        if "payload" in captured:
            page1_jobs = self._parse_api_response(captured["payload"])
            for j in page1_jobs:
                if j.id in seen_ext_ids:
                    continue
                seen_ext_ids.add(j.id)
                all_jobs.append(j)
            logger.info(f"glints: page 1 captured {len(all_jobs)} job(s)")
            await on_event(
                {
                    "type": "progress",
                    "portal": self.portal,
                    "scraped": len(all_jobs),
                    "total": len(all_jobs),
                }
            )

            api_url = captured.get("api_url")
            post_data = captured.get("post_data")
            # Forward Glints' custom request headers on replayed POSTs: x-* and
            # accept-language (required), plus authorization (the bearer token the
            # authenticated page-1 request carried — needed for login-gated pages
            # 2+). Cookies/origin/referer/UA are sent by the in-page fetch.
            extra_headers: dict[str, str] = {"content-type": "application/json"}
            for k, v in (captured.get("req_headers") or {}).items():
                kl = k.lower()
                if kl.startswith("x-") or kl in ("accept-language", "authorization"):
                    extra_headers[kl] = v

            async def _fetch_page(page_num: int) -> Any:
                # Pages 2..N via direct fetch in the browser (inherits cookies/headers).
                # Glints serves jobs from a GraphQL POST, so replay the captured POST
                # body with an incremented `page`; fall back to a GET ?page= for any
                # plain REST endpoint.
                gql_body = _graphql_page_body(post_data, page_num)
                if gql_body is not None:
                    return await page.evaluate(
                        """async ([u, body, headers]) => {
                            const r = await fetch(u, {
                                method: 'POST', credentials: 'include',
                                headers, body,
                            });
                            return await r.json();
                        }""",
                        [str(api_url), gql_body, extra_headers],
                    )
                paged_url = _with_page_param(str(api_url), page_num)
                return await page.evaluate(
                    "(u) => fetch(u, {credentials: 'include'}).then(r => r.json())",
                    paged_url,
                )

            if api_url:
                reauthed = False
                page_num = 2
                while page_num <= _MAX_PAGES and len(all_jobs) < _MAX_JOBS:
                    try:
                        data = await _fetch_page(page_num)
                    except Exception as e:
                        logger.warning(f"glints: pagination fetch failed at p{page_num}: {e}")
                        break
                    if not isinstance(data, dict):
                        break
                    page_jobs = self._parse_api_response(data)
                    if not page_jobs:
                        gated = _is_unauthenticated(data)
                        # Glints gates pages beyond the first behind login. If the
                        # session expired mid-run, re-mint it once and retry this page.
                        if gated and not reauthed:
                            reauthed = True
                            if await self._reauth(page, extra_headers):
                                logger.info(
                                    f"glints: session expired; refreshed and retrying "
                                    f"page {page_num}"
                                )
                                continue  # retry same page with the new session
                        if gated:
                            await on_event(
                                {
                                    "type": "error",
                                    "severity": "warning",
                                    "portal": self.portal,
                                    "message": (
                                        "Glints session expired or absent — returning page-1 "
                                        "results only. Re-seed it (python -m app.scrapers."
                                        "glints_login) or set GLINTS_EMAIL/GLINTS_PASSWORD."
                                    ),
                                }
                            )
                            logger.info(
                                f"glints: page {page_num} login-gated and re-auth "
                                "unavailable; stopping with page-1 results"
                            )
                        else:
                            logger.info(
                                f"glints: no jobs returned for page {page_num}; "
                                "stopping pagination (end of results)"
                            )
                        break
                    new_count = 0
                    for j in page_jobs:
                        if j.id in seen_ext_ids:
                            continue
                        seen_ext_ids.add(j.id)
                        all_jobs.append(j)
                        new_count += 1
                        if len(all_jobs) >= _MAX_JOBS:
                            break
                    logger.info(
                        f"glints: page {page_num} -> {new_count} new job(s) (total {len(all_jobs)})"
                    )
                    await on_event(
                        {
                            "type": "progress",
                            "portal": self.portal,
                            "scraped": len(all_jobs),
                            "total": min(_MAX_JOBS, len(all_jobs) + 1),
                        }
                    )
                    if new_count == 0:
                        break
                    page_num += 1
                    await asyncio.sleep(humanizer_delay("pagination"))
            try:
                return await self._enrich_with_detail(page, all_jobs, deadline=deadline)
            except asyncio.CancelledError:
                # Hard external cancellation (e.g. orchestrator wait_for hit
                # the backstop). Surface what we scraped instead of swallowing
                # 100 jobs into the empty-list return path.
                logger.warning(
                    f"glints: enrichment cancelled; returning {len(all_jobs)} "
                    "listing-only result(s)"
                )
                return all_jobs

        # Fallback: HTML scrape via BeautifulSoup
        try:
            html = await page.content()
        except Exception:
            html = ""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select('[data-cy="job-card"], a[href*="/opportunities/jobs/"]')[:30]
        for idx, card in enumerate(cards):
            title_el = card.select_one("h3, h2, [class*='title' i], [class*='Title' i]")
            company_el = card.select_one("[class*='company' i], [class*='Company' i]")
            location_el = card.select_one("[class*='location' i], [class*='city' i]")
            salary_el = card.select_one("[class*='salary' i], [class*='Salary' i]")
            title = title_el.get_text(strip=True) if title_el else ""
            company = company_el.get_text(strip=True) if company_el else ""
            location = location_el.get_text(strip=True) if location_el else "Indonesia"
            salary_txt = salary_el.get_text(strip=True) if salary_el else ""
            href = str(card.get("href") or "")
            apply_url = href if href.startswith("http") else f"https://glints.com{href}"
            ext_id = (href.rsplit("/", 1)[-1] or f"glints-fallback-{idx}").strip()
            if not title and not company:
                continue
            if ext_id in seen_ext_ids:
                continue
            seen_ext_ids.add(ext_id)
            salary_min, salary_max = _parse_html_salary(salary_txt)
            job = JobListingDTO(
                id=ext_id,
                portal="glints",
                title=title or "Unknown role",
                company=company or "Unknown",
                company_logo_bg=_logo_bg(company or "Unknown"),
                location=location,
                work_type="onsite",
                seniority=_infer_seniority(title or ""),  # type: ignore[arg-type]
                salary_min=salary_min,
                salary_max=salary_max,
                posted_date="2026-01-01",
                posted_label="recent",
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
            all_jobs.append(job)
            await on_event(
                {
                    "type": "progress",
                    "portal": self.portal,
                    "scraped": len(all_jobs),
                    "total": len(cards),
                }
            )
        try:
            return await self._enrich_with_detail(page, all_jobs, deadline=deadline)
        except asyncio.CancelledError:
            logger.warning(
                f"glints: enrichment cancelled; returning {len(all_jobs)} "
                "listing-only result(s) (HTML fallback)"
            )
            return all_jobs
