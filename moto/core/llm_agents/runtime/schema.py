from __future__ import annotations

import json
from typing import Any

from moto.core.utils import get_service_model

from ..tools.request_tools import CanonicalRequest

_MAX_DEPTH = 4


def build_full_schema(canonical: CanonicalRequest) -> str:
    """Return a JSON-formatted description of the operation's output shape for the LLM."""
    try:
        service_model = get_service_model(canonical.service)
        operation_model = service_model.operation_model(canonical.operation)
    except Exception:
        return "unavailable"

    output_shape = operation_model.output_shape
    if output_shape is None:
        return "empty_response"

    protocol = service_model.metadata.get("protocol", "unknown")
    schema = _shape_to_dict(output_shape, depth=0)
    return f"protocol={protocol}\n{json.dumps(schema, indent=2)}"


def _shape_to_dict(shape: Any, depth: int = 0, name: str = "") -> dict[str, Any]:
    if depth > _MAX_DEPTH:
        return {"type": shape.type_name}

    lowered_name = name.lower()
    result: dict[str, Any] = {"type": shape.type_name}

    # 필드명 기반 예시/형식 힌트 추가
    if "arn" in lowered_name:
        result["example"] = f"arn:aws:{shape.type_name}:region:account-id:resource"
    elif "id" in lowered_name and shape.type_name == "string":
        result["example"] = f"{lowered_name}-12345abcde"
    elif "timestamp" in lowered_name or "date" in lowered_name:
        result["example"] = "2024-05-11T12:34:56Z"

    if shape.type_name == "structure":
        required = set(getattr(shape, "required_members", []) or [])
        members: dict[str, Any] = {}
        for member_name, member_shape in shape.members.items():
            info = _shape_to_dict(member_shape, depth + 1, member_name)
            if member_name in required:
                info["required"] = True
            members[member_name] = info
        result["members"] = members
        return result

    if shape.type_name == "list":
        result["member"] = _shape_to_dict(shape.member, depth + 1, name)
        return result

    if shape.type_name == "map":
        result["key"] = shape.key.type_name
        result["value"] = _shape_to_dict(shape.value, depth + 1, name)
        return result

    if hasattr(shape, "enum") and shape.enum:
        result["enum"] = list(shape.enum)
    return result
