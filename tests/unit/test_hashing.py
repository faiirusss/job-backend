from app.utils.hashing import canonical_json, params_hash


def test_canonical_json_is_stable_across_key_order():
    a = {"role_keywords": ["python"], "location": ["jakarta"]}
    b = {"location": ["jakarta"], "role_keywords": ["python"]}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_sorts_lists_for_set_like_fields():
    a = {"role_keywords": ["python", "fastapi"], "location": ["jakarta"]}
    b = {"role_keywords": ["fastapi", "python"], "location": ["jakarta"]}
    assert canonical_json(a) == canonical_json(b)


def test_params_hash_deterministic():
    p = {"role_keywords": ["python"], "location": ["jakarta"], "work_type": ["remote"]}
    assert params_hash(p) == params_hash(p)
    assert len(params_hash(p)) == 64  # sha256 hex
