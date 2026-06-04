import hashlib
import json
from typing import Any

LIST_FIELDS_TO_SORT = {
    "role_keywords",
    "location",
    "work_type",
    "seniority",
}


def canonical_json(params: dict[str, Any]) -> str:
    normalized: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, list) and k in LIST_FIELDS_TO_SORT:
            normalized[k] = sorted(str(x) for x in v)
        else:
            normalized[k] = v
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)


def params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(params).encode("utf-8")).hexdigest()
