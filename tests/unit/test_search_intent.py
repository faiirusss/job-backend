from app.schemas import SearchParams
from app.services.search_intent import normalize_search_params


def test_normalize_search_params_removes_chat_filler_and_location_tokens():
    params = SearchParams(
        role_keywords=["tolong", "kerjaan", "laravel", "jakarta"],
        location=["Jakarta"],
    )

    normalized = normalize_search_params(params)

    assert normalized.role_keywords == ["laravel"]


def test_normalize_search_params_keeps_real_multiword_role():
    params = SearchParams(
        role_keywords=["data analyst", "jakarta"],
        location=["Jakarta"],
    )

    normalized = normalize_search_params(params)

    assert normalized.role_keywords == ["data analyst"]
