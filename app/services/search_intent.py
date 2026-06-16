import re
import unicodedata

from app.schemas import SearchParams

_WORD_RE = re.compile(r"[a-z0-9.+#-]+")

ROLE_STOP_TERMS = {
    "please",
    "tolong",
    "cari",
    "carikan",
    "kerja",
    "kerjaan",
    "pekerjaan",
    "lowongan",
    "loker",
    "job",
    "jobs",
    "role",
    "posisi",
    "di",
    "in",
    "at",
    "untuk",
    "yang",
    "dan",
    "atau",
    "remote",
    "hybrid",
    "onsite",
    "on-site",
    "wfo",
    "wfh",
    "indonesia",
}

_LOCATION_FILLER = {
    "area",
    "city",
    "dki",
    "global",
    "indonesia",
    "kabupaten",
    "kota",
    "metropolitan",
    "province",
    "remote",
}


def normalize_search_params(params: SearchParams) -> SearchParams:
    """Clean parsed intent before it reaches scrapers/cache.

    LLMs, especially the offline FakeLLM, can leak conversational filler or the
    requested city into ``role_keywords``. Portal search quality drops sharply
    when keywords become "tolong kerjaan laravel jakarta" instead of "laravel".
    """
    location_tokens = {
        token for location in params.location for token in _tokens(location) if token not in _LOCATION_FILLER
    }
    role_keywords: list[str] = []
    seen: set[str] = set()
    for raw in params.role_keywords:
        tokens = [
            token
            for token in _tokens(raw)
            if token not in ROLE_STOP_TERMS and token not in location_tokens
        ]
        if not tokens:
            continue
        keyword = " ".join(tokens)
        if keyword in seen:
            continue
        seen.add(keyword)
        role_keywords.append(keyword)

    return params.model_copy(update={"role_keywords": role_keywords})


def _tokens(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WORD_RE.findall(ascii_text.lower())
