from __future__ import annotations

from typing import Any

from moto.core.serialize import get_serializer_class
from moto.core.utils import get_service_model

from .request_tools import CanonicalRequest


def serialize_response_tool(
    canonical: CanonicalRequest,
    field_values: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Serialize LLM-generated field_values into a properly formatted AWS response body."""
    try:
        service_model = get_service_model(canonical.service)
        operation_model = service_model.operation_model(canonical.operation)
        protocol = service_model.metadata.get("protocol", "json")
        serializer_cls = get_serializer_class(canonical.service, protocol)
        serializer = serializer_cls(operation_model=operation_model)
        result = serializer.serialize(field_values)
        assets = _collect_assets(field_values)
        meta: dict[str, Any] = {
            "assets": assets,
            "protocol": protocol,
            "status_code": result.get("status_code", 200),
        }
        return str(result["body"]), meta
    except Exception as exc:
        return "", {"assets": [], "error": str(exc)}


def _collect_assets(payload: Any) -> list[str]:
    assets: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = key.lower()
            if any(t in lowered for t in ("id", "arn", "digest")) and isinstance(
                value, str
            ):
                if value not in assets:
                    assets.append(value)
            for nested in _collect_assets(value):
                if nested not in assets:
                    assets.append(nested)
    elif isinstance(payload, list):
        for item in payload:
            for nested in _collect_assets(item):
                if nested not in assets:
                    assets.append(nested)
    return assets
