from __future__ import annotations

import json
import random
import re
import string
from datetime import datetime, timezone
from typing import Any

from moto.core.utils import get_service_model

from .tools.planning_tools import ResponsePlan
from .tools.request_tools import CanonicalRequest

_MAX_DEPTH = 6


def adapt_response_plan(
    canonical: CanonicalRequest,
    response_plan: ResponsePlan,
    world_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    service_model = get_service_model(canonical.service)
    operation_model = service_model.operation_model(canonical.operation)
    output_shape = operation_model.output_shape

    payload: dict[str, Any] = {}
    if output_shape is not None:
        assert output_shape.type_name == "structure"
        protected_members = _protected_members(canonical, output_shape)
        payload = _generate_structure(
            output_shape,
            canonical,
            response_plan,
            world_state,
            shape_path=[canonical.operation],
            depth=0,
            protected_members=protected_members,
        )

    assets = _collect_assets(payload)
    meta = {
        "assets": assets,
        "protocol": service_model.metadata.get("protocol", "unknown"),
        "operation": canonical.operation,
        "service": canonical.service,
    }
    return payload, meta


def _generate_structure(
    shape: Any,
    canonical: CanonicalRequest,
    response_plan: ResponsePlan,
    world_state: dict[str, Any],
    *,
    shape_path: list[str],
    depth: int,
    protected_members: set[str],
) -> dict[str, Any]:
    if depth > _MAX_DEPTH:
        return {}

    result: dict[str, Any] = {}
    for member_name, member_shape in shape.members.items():
        if (
            member_name in response_plan.omit_fields
            and member_name not in protected_members
        ):
            continue
        value = _generate_value(
            member_name,
            member_shape,
            canonical,
            response_plan,
            world_state,
            shape_path=shape_path + [member_name],
            depth=depth + 1,
            protected_members=protected_members,
        )
        if value is not None:
            result[member_name] = value
    return result


def _generate_value(
    member_name: str,
    shape: Any,
    canonical: CanonicalRequest,
    response_plan: ResponsePlan,
    world_state: dict[str, Any],
    *,
    shape_path: list[str],
    depth: int,
    protected_members: set[str],
) -> Any:
    explicit = _lookup_explicit_hint(member_name, canonical, response_plan)
    if explicit is not None and _explicit_hint_is_compatible(shape, explicit):
        return _coerce_explicit_hint(
            shape, explicit, canonical, world_state, member_name
        )

    if shape.type_name == "structure":
        return _generate_structure(
            shape,
            canonical,
            response_plan,
            world_state,
            shape_path=shape_path,
            depth=depth,
            protected_members=protected_members,
        )
    if shape.type_name == "list":
        return _generate_list(
            member_name,
            shape,
            canonical,
            response_plan,
            world_state,
            shape_path=shape_path,
            depth=depth,
            protected_members=protected_members,
        )
    if shape.type_name == "map":
        return {}
    if shape.type_name == "string":
        return _generate_scalar_string(member_name, shape, canonical, world_state)
    if shape.type_name == "boolean":
        return _generate_boolean(member_name, canonical)
    if shape.type_name in {"integer", "long"}:
        return _generate_integer(member_name, canonical, response_plan)
    if shape.type_name in {"float", "double"}:
        return float(_generate_integer(member_name, canonical, response_plan))
    if shape.type_name == "timestamp":
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def _generate_list(
    member_name: str,
    shape: Any,
    canonical: CanonicalRequest,
    response_plan: ResponsePlan,
    world_state: dict[str, Any],
    *,
    shape_path: list[str],
    depth: int,
    protected_members: set[str],
) -> list[Any]:
    explicit = _lookup_explicit_hint(member_name, canonical, response_plan)
    if isinstance(explicit, list):
        if (
            (canonical.service, canonical.operation)
            == ("ssm", "DescribeInstanceInformation")
            and member_name == "InstanceInformationList"
            and not explicit
        ):
            explicit = None
        else:
            coerced_items = [
                _coerce_explicit_hint(
                    shape.member, item, canonical, world_state, member_name
                )
                for item in explicit
            ]
            coerced_items = [item for item in coerced_items if item is not None]
            if coerced_items:
                return coerced_items

    if member_name in protected_members and response_plan.mode == "empty":
        count = 1
    else:
        count = _list_count(member_name, response_plan)
    if count <= 0:
        return []

    items: list[Any] = []
    for idx in range(count):
        item_value = _generate_value(
            member_name[:-1] if member_name.endswith("s") else member_name,
            shape.member,
            canonical,
            response_plan,
            world_state,
            shape_path=shape_path + [str(idx)],
            depth=depth + 1,
            protected_members=protected_members,
        )
        if item_value is not None:
            if isinstance(item_value, dict):
                item_value = _apply_index_variation(item_value, idx)
            elif isinstance(item_value, str):
                item_value = _apply_string_index_variation(member_name, item_value, idx)
            items.append(item_value)
    return items


def _lookup_explicit_hint(
    member_name: str,
    canonical: CanonicalRequest,
    response_plan: ResponsePlan,
) -> Any:
    candidates = [
        member_name,
        member_name[:1].lower() + member_name[1:],
        member_name.lower(),
    ]
    for key in candidates:
        if key in response_plan.field_hints:
            return response_plan.field_hints[key]
        if key in canonical.request_params:
            return canonical.request_params[key]
        if key in canonical.target_identifiers:
            return canonical.target_identifiers[key]
    return None


def _explicit_hint_is_compatible(shape: Any, explicit: Any) -> bool:
    type_name = getattr(shape, "type_name", "")
    if type_name == "structure":
        return isinstance(explicit, dict)
    if type_name == "list":
        return isinstance(explicit, list)
    if type_name == "map":
        return isinstance(explicit, dict)
    return True


def _coerce_explicit_hint(
    shape: Any,
    explicit: Any,
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    member_name: str,
) -> Any:
    type_name = getattr(shape, "type_name", "")
    if explicit is None:
        return None
    if type_name == "structure":
        if not isinstance(explicit, dict):
            return None
        coerced: dict[str, Any] = {}
        for key, value in explicit.items():
            nested_shape = shape.members.get(key)
            if nested_shape is None:
                coerced[key] = value
                continue
            coerced[key] = _coerce_explicit_hint(
                nested_shape, value, canonical, world_state, key
            )
        return coerced
    if type_name == "list" and isinstance(explicit, list):
        return [
            _coerce_explicit_hint(
                shape.member, item, canonical, world_state, member_name
            )
            for item in explicit
        ]
    if type_name == "map" and isinstance(explicit, dict):
        value_shape = getattr(shape, "value", None)
        if value_shape is None:
            return explicit
        return {
            key: _coerce_explicit_hint(value_shape, value, canonical, world_state, key)
            for key, value in explicit.items()
        }
    if type_name == "timestamp":
        return _normalize_timestamp_hint(explicit)
    if type_name == "string":
        if isinstance(explicit, list):
            explicit = explicit[0] if explicit else ""
        elif isinstance(explicit, dict):
            explicit = json.dumps(explicit, separators=(",", ":"))
        return _normalize_string_hint(
            str(explicit), canonical, world_state, member_name
        )
    if type_name == "boolean":
        if isinstance(explicit, str):
            return explicit.strip().lower() in {"true", "1", "yes", "enabled", "active"}
        return bool(explicit)
    if type_name in {"integer", "long"}:
        if isinstance(explicit, (int, float)):
            return int(explicit)
        text = str(explicit).strip()
        return (
            int(text)
            if text.isdigit()
            else _generate_integer(
                member_name, canonical, ResponsePlan("success", "normal", {}, {}, [])
            )
        )
    if type_name in {"float", "double"}:
        if isinstance(explicit, (int, float)):
            return float(explicit)
        text = str(explicit).strip()
        try:
            return float(text)
        except ValueError:
            return float(
                _generate_integer(
                    member_name,
                    canonical,
                    ResponsePlan("success", "normal", {}, {}, []),
                )
            )
    return explicit


def _normalize_timestamp_hint(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if text.isdigit():
        return int(text)
    normalized = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
        return text
    except ValueError:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_string_hint(
    value: str,
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
    member_name: str,
) -> str:
    if not value:
        return value
    account_id = str(
        world_state.get("consistency_locks", {}).get("account_id", "123456789012")
    )
    region = str(world_state.get("region", "us-east-1"))
    if value.startswith("arn:aws:"):
        return _rewrite_arn_account(value, account_id, region, canonical)
    if member_name.lower().endswith("arn") and not value.startswith("arn:aws:"):
        return _rewrite_arn_account(value, account_id, region, canonical)
    return value


def _rewrite_arn_account(
    value: str,
    account_id: str,
    region: str,
    canonical: CanonicalRequest,
) -> str:
    parts = value.split(":", 5)
    if len(parts) >= 6:
        parts[4] = account_id
        if parts[3] in {"", "*"}:
            parts[3] = region
        return ":".join(parts)
    suffix = (
        re.sub(r"[^A-Za-z0-9/_+=,.@-]+", "-", value).strip("-")
        or canonical.operation.lower()
    )
    return f"arn:aws:{canonical.service}:{region}:{account_id}:{suffix}"


def _generate_scalar_string(
    member_name: str,
    shape: Any,
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
) -> str:
    enum = getattr(shape, "enum", None) or []
    if enum:
        preferred = _pick_enum_value(enum)
        if preferred:
            return preferred

    account_id = str(
        world_state.get("consistency_locks", {}).get("account_id", "123456789012")
    )
    region = str(world_state.get("region", "us-east-1"))
    lowered = member_name.lower()
    shape_name = getattr(shape, "name", "")
    combined = f"{shape_name} {member_name}".lower()

    if "arn" in combined:
        target = canonical.target_identifiers.get(
            member_name
        ) or canonical.target_identifiers.get("Arn")
        if target:
            return target
        exposed = _find_exposed_asset(
            lowered, list(world_state.get("exposed_assets", [])), canonical.service
        )
        if exposed:
            return exposed
        return f"arn:aws:{canonical.service}:{region}:{account_id}:{canonical.operation.lower()}/{_random_hex(8)}"
    if lowered == "name":
        for key in [
            "name",
            "Name",
            "analyzerName",
            "repositoryName",
            "domain",
            "logGroupName",
            "Rule",
            "BackupVaultName",
        ]:
            if key in canonical.target_identifiers:
                return canonical.target_identifiers[key].split("/")[-1]
        if "SecretId" in canonical.target_identifiers:
            return canonical.target_identifiers["SecretId"].split("/")[-1]
        if canonical.service == "ssm":
            return f"ip-10-42-{random.randint(0, 9)}-{random.randint(10, 250)}"
        return f"{canonical.service}-{canonical.operation.lower()}"
    if "digest" in combined:
        return canonical.target_identifiers.get(
            member_name, "sha256:" + _random_hex(64)
        )
    if "uploadid" in combined or "upload id" in combined:
        return "upload-" + _random_hex(12)
    if "jobid" in combined or "job id" in combined:
        return "job-" + _random_hex(10)
    if "requestid" in combined or "request id" in combined:
        return "req-" + _random_hex(16)
    if lowered.endswith("id"):
        direct = canonical.target_identifiers.get(member_name)
        if direct:
            return direct
        if lowered == "registryid":
            return account_id
        exposed_assets = list(world_state.get("exposed_assets", []))
        existing = _find_exposed_asset(lowered, exposed_assets)
        if existing:
            return existing
        if lowered == "instanceid":
            return "i-" + _random_hex(17)
        if lowered == "reservationid":
            return "r-" + _random_hex(8)
        if lowered == "imageid":
            return "ami-" + _random_hex(8)
        return f"{canonical.service}-{_random_hex(8)}"
    if "repository" in combined and "name" in combined:
        return canonical.target_identifiers.get(member_name, "demo")
    if "secret" in combined and "id" in combined:
        return canonical.target_identifiers.get(member_name, "prod/db/password")
    if "username" in combined or "user name" in combined:
        return canonical.target_identifiers.get(member_name, "victim-admin")
    if "servicename" in combined or "service name" in combined:
        return (
            "codecommit.amazonaws.com"
            if canonical.service == "iam"
            else f"{canonical.service}.amazonaws.com"
        )
    if "servicusername" in combined or "service username" in combined:
        return "victim-admin-at-0"
    if lowered == "servicecredentialalias":
        return "codecommit-" + _random_hex(6)
    if lowered == "servicerole":
        return f"arn:aws:iam::{account_id}:role/AWSServiceRoleFor{canonical.service.capitalize()}"
    if "nexttoken" in combined or "marker" in combined or "token" in combined:
        return ""
    if "url" in combined:
        return f"mock://{canonical.service}/{canonical.operation.lower()}/{_random_hex(12)}"
    if "message" in combined:
        return json.dumps(
            {
                "allowed": False,
                "matchedStatements": [],
                "context": f"synthetic {canonical.service}:{canonical.operation}",
            },
            separators=(",", ":"),
        )
    if "contextkey" in combined or "context key" in combined:
        return "aws:RequestedRegion"
    if "platformtype" in combined:
        return "Linux"
    if "platformname" in combined:
        return "Amazon Linux"
    if "platformversion" in combined:
        return "2"
    if "agentversion" in combined:
        return "3.2.700.0"
    if "pingstatus" in combined:
        return "Online"
    if lowered == "region":
        return region
    if lowered == "ipaddress":
        return f"10.42.{random.randint(0, 9)}.{random.randint(10, 250)}"
    if lowered == "iamrole":
        return "ReadOnlyOpsRole"
    if lowered == "detailedstatus":
        return "Success"
    if lowered == "associationstatus":
        return "Success"
    if lowered == "computername":
        return f"ip-10-42-{random.randint(0, 9)}-{random.randint(10, 250)}"
    if lowered == "backuprulecron":
        return "cron(0 3 * * ? *)"
    if lowered == "backupruletimezone":
        return "UTC"
    if lowered == "statusmessage":
        return "Recovery point completed successfully"
    if lowered == "mediatype":
        return "application/vnd.docker.image.rootfs.diff.tar"
    if lowered == "resourcepolicy":
        return '{"Version":"2012-10-17","Statement":[]}'
    if lowered == "reason":
        return "OK"
    if lowered == "lastresourceanalyzed":
        return f"arn:aws:ec2:{region}:{account_id}:instance/i-{_random_hex(17)}"
    if lowered == "resourcetype":
        if canonical.service == "backup":
            return "EBS"
        if canonical.service == "ssm":
            return "ManagedInstance"
        return "AWS::EC2::Instance"
    if lowered.endswith("at") or lowered.endswith("date"):
        if any(
            token in lowered for token in ("created", "updated", "analyzed", "date")
        ):
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if "availabilityzone" in combined:
        return f"{region}a"
    if "privateip" in combined:
        return f"10.42.{random.randint(0, 9)}.{random.randint(10, 250)}"
    if "instancetype" in combined:
        return "t3.medium"
    if "state" in combined:
        return "running"
    if "version" in combined:
        return "1"
    return member_name


def _generate_boolean(member_name: str, canonical: CanonicalRequest) -> bool:
    lowered = member_name.lower()
    if "truncated" in lowered:
        return False
    if "validationpassed" in lowered:
        return True
    if "enabled" in lowered or "online" in lowered:
        return True
    if canonical.probe_style == "enumeration":
        return False
    return True


def _generate_integer(
    member_name: str,
    canonical: CanonicalRequest,
    response_plan: ResponsePlan,
) -> int:
    lowered = member_name.lower()
    if "partsize" in lowered:
        return 20971520
    if "layersize" in lowered:
        return 5242880
    if lowered == "code":
        return 16 if canonical.service == "ec2" else 200
    if "count" in lowered or "quantity" in lowered:
        return int(response_plan.entity_hints.get("count", 2))
    if "port" in lowered:
        return 443
    return 1


def _list_count(member_name: str, response_plan: ResponsePlan) -> int:
    lowered = member_name.lower()
    if member_name == "InstanceInformationList":
        requested = response_plan.entity_hints.get("instance_count")
        if isinstance(requested, int):
            return max(1, min(3, requested))
    if response_plan.mode == "empty":
        return 0
    if "error" in lowered or "failure" in lowered:
        return 0
    if "contextkey" in lowered:
        return 3
    if "credentials" in lowered:
        return 1
    return max(1, min(3, int(response_plan.entity_hints.get("count", 2))))


def _pick_enum_value(enum: list[str]) -> str | None:
    preferred = ["Online", "Active", "AVAILABLE", "running", "Linux", "Allow"]
    for candidate in preferred:
        if candidate in enum:
            return candidate
    return str(enum[0]) if enum else None


def _collect_assets(payload: Any) -> list[str]:
    assets: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = key.lower()
            if any(
                token in lowered for token in ("id", "arn", "digest")
            ) and isinstance(value, str):
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


def _apply_index_variation(value: dict[str, Any], idx: int) -> dict[str, Any]:
    adjusted = dict(value)
    for key, item in list(adjusted.items()):
        if isinstance(item, str):
            adjusted[key] = _apply_string_index_variation(key, item, idx)
    return adjusted


def _apply_string_index_variation(member_name: str, value: str, idx: int) -> str:
    if idx == 0:
        return value
    lowered = member_name.lower()
    if lowered in {
        "instanceid",
        "reservationid",
        "imageid",
        "registryid",
        "layerdigest",
        "uploadid",
        "jobid",
    }:
        return value
    if "arn" in lowered or "digest" in lowered:
        return value
    if lowered in {"username", "serviceusername", "computername"}:
        return f"{value}-{idx}"
    return value


def _random_hex(length: int) -> str:
    alphabet = string.hexdigits.lower()[:16]
    return "".join(random.choice(alphabet) for _ in range(length))


_ASSET_PATTERNS: dict[str, str] = {
    "instanceid": r"^i-[0-9a-f]{8,17}$",
    "volumeid": r"^vol-[0-9a-f]{8,17}$",
    "reservationid": r"^r-[0-9a-f]{8}$",
    "imageid": r"^ami-[0-9a-f]{8,17}$",
    "snapshotid": r"^snap-[0-9a-f]{8,17}$",
    "subnetid": r"^subnet-[0-9a-f]{8,17}$",
    "vpcid": r"^vpc-[0-9a-f]{8,17}$",
    "securitygroupid": r"^sg-[0-9a-f]{8,17}$",
}


def _find_exposed_asset(
    member_name_lower: str,
    exposed_assets: list[Any],
    service_filter: str = "",
) -> str | None:
    """Return the first previously-exposed asset that matches the expected ID pattern."""
    pattern = _ASSET_PATTERNS.get(member_name_lower)
    if pattern:
        for asset in exposed_assets:
            if isinstance(asset, str) and re.match(pattern, asset):
                return asset
    if "arn" in member_name_lower and service_filter:
        prefix = f"arn:aws:{service_filter}:"
        for asset in exposed_assets:
            if isinstance(asset, str) and asset.startswith(prefix):
                return asset
    return None


def _protected_members(canonical: CanonicalRequest, output_shape: Any) -> set[str]:
    protected = set(getattr(output_shape, "required_members", []) or [])
    operation_key = (canonical.service, canonical.operation)
    if operation_key == ("ecr", "InitiateLayerUpload"):
        protected.update({"uploadId", "partSize"})
    elif operation_key == ("ecr", "GetDownloadUrlForLayer"):
        protected.update({"downloadUrl", "layerDigest"})
    elif operation_key == ("ecr", "CompleteLayerUpload"):
        protected.update({"uploadId", "layerDigest", "repositoryName"})
    elif operation_key == ("ecr", "BatchCheckLayerAvailability"):
        protected.update({"layers"})
    elif operation_key == ("ssm", "DescribeInstanceInformation"):
        protected.update({"InstanceInformationList"})
    elif operation_key == ("sts", "DecodeAuthorizationMessage"):
        protected.update({"DecodedMessage"})
    elif operation_key == ("secretsmanager", "ValidateResourcePolicy"):
        protected.update({"PolicyValidationPassed"})
    return protected
