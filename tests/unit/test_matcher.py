from app.ai.matcher import cosine_score_pct, hybrid_score


def test_cosine_score_pct_identical_vectors_is_100():
    v = [1.0, 0.0, 0.0]
    assert cosine_score_pct(v, v) == 100


def test_cosine_score_pct_opposite_vectors_is_0():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_score_pct(a, b) == 0


def test_cosine_score_pct_orthogonal_is_50():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_score_pct(a, b) == 50


def test_hybrid_score_uses_60_40_weights():
    # cosine = 100, llm = 50 → 0.6*100 + 0.4*50 = 80
    assert hybrid_score(cosine_pct=100, llm_score=50) == 80


def test_hybrid_score_rounds_to_int():
    # cosine = 80, llm = 75 → 0.6*80 + 0.4*75 = 48 + 30 = 78
    assert hybrid_score(cosine_pct=80, llm_score=75) == 78
