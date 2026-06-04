import re
from hashlib import sha1

from app.ai.llm import CoverLetterPair, JDExtraction, MatchOutput
from app.schemas import JobListingDTO, SearchParams

# Keyword cues used by the FakeLLM stand-in to bucket description lines without a
# real model. Production extraction uses Gemini (see GeminiLLM.extract_jd_fields);
# this only needs to be deterministic for local/offline runs and tests.
_FAKE_MAND_KW = (
    "requirement",
    "qualification",
    "kualifikasi",
    "persyaratan",
    "syarat",
    "wajib",
    "must",
    "minimal",
)
_FAKE_NICE_KW = (
    "nice to have",
    "nice-to-have",
    "preferred",
    "bonus",
    "plus",
    "diutamakan",
    "lebih disukai",
    "nilai tambah",
)
_FAKE_BENEFIT_KW = (
    "benefit",
    "insurance",
    "asuransi",
    "allowance",
    "tunjangan",
    "bpjs",
    "cuti",
    "annual leave",
)

_ID_HINTS = {"yang", "di", "untuk", "loker", "gaji", "atau", "dan", "juta", "cari"}
_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "in",
    "at",
    "to",
    "of",
    "a",
    "an",
    "or",
    "is",
    "are",
    "be",
    "by",
    "on",
    "as",
    "this",
    "that",
    "from",
    "cari",
    "loker",
    "gaji",
    "minimal",
    "yang",
    "atau",
    "dan",
    "untuk",
    "di",
    "juta",
}

_CL_TEMPLATE_ID = """Yth. Tim HRD {company},

Saya menulis surat ini untuk menyatakan minat saya terhadap posisi {title} di {company}.
Latar belakang saya sebagai engineer dengan pengalaman menggunakan {skills} membuat saya
yakin dapat memberikan kontribusi yang berarti pada tim Anda.

Selama karier saya, saya telah mengerjakan berbagai proyek yang mencakup pengembangan
backend, integrasi sistem, serta optimasi performa. Saya juga terbiasa bekerja dalam
tim lintas-fungsi, menjaga kualitas kode, dan berkolaborasi melalui pull request review.

Saya tertarik secara khusus pada {company} karena reputasinya dalam membangun produk yang
berdampak. Saya ingin sekali berkontribusi pada {title} dan terus berkembang bersama tim.

Saya bersedia mendiskusikan lebih lanjut bagaimana pengalaman saya selaras dengan kebutuhan
{company}. Saya dapat dihubungi melalui email kapan saja.

Hormat saya,
Kandidat
""".strip()

_CL_TEMPLATE_EN = """Dear Hiring Manager,

I am writing to express my interest in the {title} role at {company}. My background as
an engineer with hands-on experience in {skills} aligns closely with the requirements
outlined in your description, and I am confident I can contribute meaningfully to your team.

Over my career, I have delivered backend services end-to-end, integrated third-party
systems, and continuously improved code quality through reviews and pair work. I value
clean architecture, observable systems, and pragmatic trade-offs.

I am especially drawn to {company} because of its reputation for building impactful
products. I would welcome the chance to contribute to the {title} team and grow alongside
talented engineers.

I would be glad to discuss how my experience matches your needs in more detail and am
available to talk at your convenience.

Sincerely,
The Candidate
""".strip()


class FakeLLM:
    async def parse_intent(self, query: str) -> SearchParams:
        q = query.lower()
        work_type: list = []
        for wt in ("remote", "hybrid", "onsite"):
            if wt in q:
                work_type.append(wt)
        if not work_type:
            work_type = ["remote", "hybrid", "onsite"]

        salary_min = None
        m = re.search(r"(\d+)\s*juta", q)
        if m:
            salary_min = int(m.group(1)) * 1_000_000

        tokens = [t.strip(".,!?") for t in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{2,}", query)]
        role_keywords = [
            t
            for t in tokens
            if t.lower() not in _STOP_WORDS and t.lower() not in {"remote", "hybrid", "onsite"}
        ][:6]

        locations: list[str] = []
        for city in ("Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Medan", "Bali"):
            if city.lower() in q:
                locations.append(city)
        if "remote" in q:
            locations.append("Remote-Indonesia")
        if not locations:
            locations = ["Indonesia"]

        return SearchParams(
            role_keywords=role_keywords,
            location=locations,
            work_type=work_type,
            seniority=None,
            salary_min_idr=salary_min,
        )

    async def score_jobs(self, cv_text: str, jobs: list[JobListingDTO]) -> list[MatchOutput]:
        cv_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{2,}", cv_text.lower()))
        results: list[MatchOutput] = []
        for job in jobs:
            seed = int(sha1(job.id.encode()).hexdigest(), 16)
            llm_score = 60 + (seed % 30)
            corpus = " ".join(
                [
                    job.description or "",
                    " ".join(job.responsibilities),
                    " ".join(job.mandatory_requirements),
                    " ".join(job.nice_to_have_requirements),
                    " ".join(job.skills_tags),
                ]
            )
            desc_tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{2,}", corpus.lower())
            matched: list[str] = []
            missing: list[str] = []
            seen: set[str] = set()
            for t in desc_tokens:
                if t in seen:
                    continue
                seen.add(t)
                if t in cv_tokens and len(matched) < 5:
                    matched.append(t)
                elif t not in cv_tokens and len(missing) < 5 and len(t) >= 3:
                    missing.append(t)
                if len(matched) >= 5 and len(missing) >= 5:
                    break
            results.append(
                MatchOutput(
                    llm_score=llm_score,
                    matched_skills=matched,
                    missing_skills=missing,
                    summary_id=f"Cocok untuk peran {job.title} di {job.company}.",
                    summary_en=f"Decent fit for {job.title} at {job.company}.",
                )
            )
        return results

    async def extract_jd_fields(self, jobs: list[JobListingDTO]) -> list[JDExtraction]:
        """Deterministic stand-in for Gemini's structured-data cleaner: buckets
        description lines by keyword cue. Heading-only lines are dropped; lines
        with no cue fall into responsibilities. skills_tags are left to the
        caller's union with the scraper's structured array."""
        out: list[JDExtraction] = []
        for job in jobs:
            resp: list[str] = []
            mand: list[str] = []
            nice: list[str] = []
            benefits: list[str] = []
            for raw in (job.description or "").split("\n"):
                line = re.sub(r"^[-•*·◦▪\d.)\s]+", "", raw).strip()
                if not line:
                    continue
                low = line.lower()
                if any(k in low for k in _FAKE_MAND_KW):
                    bucket = mand
                elif any(k in low for k in _FAKE_NICE_KW):
                    bucket = nice
                elif any(k in low for k in _FAKE_BENEFIT_KW):
                    bucket = benefits
                else:
                    bucket = resp
                # Treat a short cue-bearing line ("Requirements:", "Nilai Plus")
                # as a heading and skip it rather than emitting it as an item.
                if bucket is not resp and len(line.split()) <= 3:
                    continue
                bucket.append(line)
            out.append(
                JDExtraction(
                    responsibilities=resp,
                    mandatory_requirements=mand,
                    nice_to_have_requirements=nice,
                    skills_tags=[],
                    benefits=benefits,
                )
            )
        return out

    async def generate_intro(self, query: str, params: SearchParams) -> str:
        role = params.role_keywords[0] if params.role_keywords else "pekerjaan"
        loc = ", ".join(params.location) if params.location else "Indonesia"
        portals = "Glints"  # MVP scrapes Glints only
        wt_label_map = {"remote": "remote", "hybrid": "hybrid", "onsite": "on-site"}
        wt = (
            ", ".join(wt_label_map.get(w, w) for w in params.work_type)
            if params.work_type
            else "semua tipe"
        )
        return (
            f"Baik, saya akan mencarikan lowongan untuk posisi {role} "
            f"di {loc} ({wt}) via {portals}. Tunggu sebentar ya."
        )

    async def generate_cover_letter(
        self, cv_text: str, job: JobListingDTO, matched_skills: list[str]
    ) -> CoverLetterPair:
        skills_str = ", ".join(matched_skills) if matched_skills else "modern web technologies"
        id_text = _CL_TEMPLATE_ID.format(company=job.company, title=job.title, skills=skills_str)
        en_text = _CL_TEMPLATE_EN.format(company=job.company, title=job.title, skills=skills_str)
        return CoverLetterPair(
            content_id=id_text,
            content_en=en_text,
            word_count_id=len(id_text.split()),
            word_count_en=len(en_text.split()),
        )
