"""Portal-agnostic, pure helpers shared by scraper normalizers.

Indonesian enum-label maps + small deterministic helpers. No I/O, no network.
Lifted out of glints_normalize.py / glints.py so the LinkedIn normalizer reuses
them without importing across scraper modules.
"""

import hashlib

JOB_TYPE_LABEL = {
    "FULL_TIME": "Penuh Waktu",
    "PART_TIME": "Paruh Waktu",
    "INTERNSHIP": "Magang",
    "PROJECT_BASED": "Freelance",
    "CONTRACT": "Kontrak",
}
WORK_ARRANGEMENT_LABEL = {
    "ONSITE": "Kerja di lokasi",
    "REMOTE": "Remote / dari rumah",
    "HYBRID": "Hybrid",
}
EDUCATION_LABEL = {
    "PRIMARY_SCHOOL": "SD",
    "SECONDARY_SCHOOL": "SMP",
    "HIGH_SCHOOL": "SMA/SMK",
    "DIPLOMA": "Diploma (D1–D4)",
    "COLLEGE_DEGREE": "Diploma III",
    "BACHELOR_DEGREE": "Sarjana (S1)",
    "PROFESSIONAL_EDUCATION": "Pendidikan Profesi",
    "MASTER_DEGREE": "Magister (S2)",
    "DOCTORATE": "Doktor (S3)",
}
COMPANY_SIZE_LABEL = {
    "SELF_EMPLOYED": "Wiraswasta",
    "BETWEEN_1_AND_10": "1–10 karyawan",
    "BETWEEN_11_AND_50": "11–50 karyawan",
    "BETWEEN_51_AND_200": "51–200 karyawan",
    "BETWEEN_201_AND_500": "201–500 karyawan",
    "BETWEEN_501_AND_1000": "501–1000 karyawan",
    "BETWEEN_1001_AND_5000": "1001–5000 karyawan",
    "BETWEEN_5001_AND_10000": "5001–10.000 karyawan",
    "MORE_THAN_10000": "> 10.000 karyawan",
}

_LOGO_PALETTE = [
    "#6366f1",
    "#ec4899",
    "#10b981",
    "#f59e0b",
    "#3b82f6",
    "#8b5cf6",
    "#ef4444",
    "#14b8a6",
    "#f97316",
    "#06b6d4",
]


def enum_label(value: str | None, mapping: dict[str, str]) -> str:
    if not value:
        return ""
    return mapping.get(value) or value.replace("_", " ").title()


def experience_label(mn: int | None, mx: int | None) -> str:
    mn = mn or 0
    mx = mx or 0
    if mn == 0 and mx <= 1:
        return "Kurang dari 1 tahun / fresh graduate"
    if mx and mx >= 10:
        return "Lebih dari 10 tahun"
    return f"{mn}–{mx} tahun pengalaman"


def logo_bg(company: str) -> str:
    h = int(hashlib.sha1(company.encode("utf-8")).hexdigest(), 16)
    return _LOGO_PALETTE[h % len(_LOGO_PALETTE)]


def map_work_type(raw: str | None) -> str:
    if not raw:
        return "onsite"
    s = raw.strip().lower()
    if "remote" in s:
        return "remote"
    if "hybrid" in s:
        return "hybrid"
    return "onsite"


def infer_seniority(title: str) -> str:
    t = title.lower()
    if any(w in t for w in ("senior", "lead", "principal", "staff")):
        return "senior"
    if any(w in t for w in ("junior", "jr.", "intern", "associate")):
        return "junior"
    return "mid"


def posted_label(iso_date: str | None) -> str:
    if not iso_date:
        return "recent"
    return iso_date[:10]
