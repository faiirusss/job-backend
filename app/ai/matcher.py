import math


def cosine_score_pct(cv_emb: list[float], job_emb: list[float]) -> int:
    if not cv_emb or not job_emb or len(cv_emb) != len(job_emb):
        return 0
    dot = sum(a * b for a, b in zip(cv_emb, job_emb, strict=True))
    na = math.sqrt(sum(a * a for a in cv_emb))
    nb = math.sqrt(sum(b * b for b in job_emb))
    if na == 0.0 or nb == 0.0:
        return 0
    cos = dot / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    pct = (cos + 1.0) / 2.0 * 100.0
    return round(pct)


def hybrid_score(cosine_pct: int, llm_score: int) -> int:
    return round(0.6 * cosine_pct + 0.4 * llm_score)
