from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from moto.core.utils import get_service_model

from .tools.planning_tools import ResponsePlan
from .tools.request_tools import CanonicalRequest
from .tools.state_tools import _param_cache_key

_MAX_DEPTH = 6


def adapt_response_plan(
    canonical: CanonicalRequest,
    response_plan: ResponsePlan,
    world_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    service_model = get_service_model(canonical.service)
    operation_model = service_model.operation_model(canonical.operation)
    output_shape = operation_model.output_shape
    protocol = service_model.metadata.get("protocol", "unknown")

    payload: dict[str, Any] = {}

    # 알고 있는 native 페이지네이션 토큰이 포함된 요청은 캐시 우회 → NextToken 없는 새 응답 생성
    seen_tokens = world_state.get("seen_pagination_tokens", [])
    is_paginated_continuation = any(
        isinstance(v, str) and v in seen_tokens
        for v in canonical.request_params.values()
    )

    # Return cached field_values for repeated read calls to guarantee consistency
    cached = world_state.get("response_cache", {}).get(
        _param_cache_key(
            canonical.service, canonical.operation, canonical.target_identifiers
        )
    )
    if cached is not None and not is_paginated_continuation:
        payload = deepcopy(cached)
        meta = {
            "assets": _collect_assets(payload),
            "protocol": protocol,
            "operation": canonical.operation,
            "service": canonical.service,
        }
        return payload, meta

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
        "protocol": protocol,
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
    # botocore structure member들을 list로 바꿔 union shape에서 일부만 선택할 수 있게 한다.
    members = list(shape.members.items())
    # union shape는 동시에 여러 member가 있으면 AWS CLI parser가 실패하므로 하나만 생성한다.
    if getattr(shape, "metadata", {}).get("union"):
        members = members[:1]
    # 선택된 member들에 대해 fallback 응답 값을 생성한다.
    for member_name, member_shape in members:
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
    # LLM response_plan, request params, target identifiers에서 같은 필드를 여러 표기법으로 찾는다.
    candidates = [
        # botocore output member 이름 그대로 찾는다.
        member_name,
        # Query protocol의 list 첫 항목 표기인 InstanceId.1 같은 key를 찾는다.
        f"{member_name}.1",
        # lowerCamelCase 표기로 들어온 hint를 찾는다.
        member_name[:1].lower() + member_name[1:],
        # lowerCamelCase + .1 표기로 들어온 query hint를 찾는다.
        f"{member_name[:1].lower() + member_name[1:]}.1",
        # 완전 소문자 표기로 들어온 hint를 찾는다.
        member_name.lower(),
        # 완전 소문자 + .1 표기로 들어온 query hint를 찾는다.
        f"{member_name.lower()}.1",
    ]
    # 우선 LLM이 명시한 field_hints를 가장 신뢰한다.
    for key in candidates:
        if key in response_plan.field_hints:
            return response_plan.field_hints[key]
        # 다음으로 실제 요청 파라미터에서 온 값을 반영한다.
        if key in canonical.request_params:
            return canonical.request_params[key]
        # 마지막으로 identifier extractor가 뽑아둔 값을 반영한다.
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
    # 빈 문자열은 의미를 바꿀 필요가 없으므로 그대로 둔다.
    if not value:
        return value
    # ARN/account 보정에 사용할 세션 고정 account id를 읽는다.
    account_id = str(
        world_state.get("consistency_locks", {}).get("account_id", "123456789012")
    )
    # ARN/region 보정에 사용할 현재 region을 읽는다.
    region = str(world_state.get("region", "us-east-1"))
    # 이미 ARN이면 account/region이 세션과 일치하도록 다시 쓴다.
    if value.startswith("arn:aws:"):
        return _rewrite_arn_account(value, account_id, region, canonical)
    # ARN 필드인데 ARN 형식이 아니면 plausible ARN으로 변환한다.
    if member_name.lower().endswith("arn") and not value.startswith("arn:aws:"):
        return _rewrite_arn_account(value, account_id, region, canonical)
    # member 이름 비교를 위해 소문자로 정규화한다.
    lowered = member_name.lower()

    # ── 리소스 ID 재작성 규칙 ──────────────────────────────────────────────────
    # LLM이 제공한 generic 예시 ID를 세션+리소스 고유값으로 재작성한다.
    # 규칙: (1) 요청에서 명시한 리소스 ID → 그대로 echo (실제 AWS 동작)
    #       (2) 그 외 LLM 생성 ID → account_id+value 기반 deterministic 재작성
    #       이렇게 하면: 다른 리소스 쿼리 → 다른 ID, 다른 세션 → 다른 ID
    def _requested_id(prefix: str) -> str | None:
        # target_identifiers 먼저 (이미 정규화된 식별자)
        for k, v in canonical.target_identifiers.items():
            if k.lower() == lowered and isinstance(v, str) and v.startswith(prefix):
                return v
        # request_params도 검색 — EC2 쿼리 프로토콜의 VolumeId.1 형식 처리
        for candidate in [member_name, f"{member_name}.1", lowered, f"{lowered}.1"]:
            pv = canonical.request_params.get(candidate)
            if isinstance(pv, str) and pv.startswith(prefix):
                return pv
        return None

    if lowered == "instanceid":
        requested = _requested_id("i-")
        if requested:
            return requested  # 요청에서 명시한 instance ID echo
        return "i-" + _det_hex(f"{account_id}:{value}", "instanceid", 17)
    if lowered == "volumeid":
        requested = _requested_id("vol-")
        if requested:
            return requested
        return "vol-" + _det_hex(f"{account_id}:{value}", "volumeid", 17)
    if lowered == "vpcid":
        requested = _requested_id("vpc-")
        if requested:
            return requested
        return "vpc-" + _det_hex(f"{account_id}:{value}", "vpcid", 17)
    if lowered == "subnetid":
        requested = _requested_id("subnet-")
        if requested:
            return requested
        return "subnet-" + _det_hex(f"{account_id}:{value}", "subnetid", 17)
    if lowered in {"groupid", "securitygroupid"}:
        requested = _requested_id("sg-")
        if requested:
            return requested
        return "sg-" + _det_hex(f"{account_id}:{value}", "securitygroupid", 17)
    # ReservedInstancesId는 UUID 형식 (rsvd- 등 잘못된 prefix 보정)
    if lowered == "reservedinstancesid" and not re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value
    ):
        h = _det_hex(value, "reservedinstancesid", 32)
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    # uploadId는 UUID 형식 (hex only, "example"·"demo" 등 placeholder 보정)
    if lowered == "uploadid" and not re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value
    ):
        h = _det_hex(f"{account_id}:{value}", "uploadid", 32)
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    return value


def _rewrite_arn_account(
    value: str,
    account_id: str,
    region: str,
    canonical: CanonicalRequest,
) -> str:
    parts = value.split(":", 5)
    if len(parts) < 6:
        suffix = (
            re.sub(r"[^A-Za-z0-9/_+=,.@-]+", "-", value).strip("-")
            or canonical.operation.lower()
        )
        return f"arn:aws:{canonical.service}:{region}:{account_id}:{suffix}"

    arn_service = parts[2]  # arn:aws:<service>:region:account:resource

    # IAM/STS ARN: region 없음, account 있음  arn:aws:iam::123456789012:...
    if arn_service in {"iam", "sts"}:
        parts[3] = ""
        parts[4] = account_id
    # Bedrock foundation-model ARN: region 있음, account 없음
    elif (
        arn_service == "bedrock"
        and len(parts) >= 6
        and parts[5].startswith("foundation-model")
    ):
        parts[3] = region if not parts[3] or parts[3] in {"*"} else parts[3]
        parts[4] = ""
    # S3 ARN: region 없음, account 없음
    elif arn_service == "s3":
        parts[3] = ""
        parts[4] = ""
    else:
        # 기본: account 설정, region이 비어있으면 현재 region으로 채움
        parts[4] = account_id
        if parts[3] in {"", "*"}:
            parts[3] = region

    return ":".join(parts)


def _generate_scalar_string(
    member_name: str,
    shape: Any,
    canonical: CanonicalRequest,
    world_state: dict[str, Any],
) -> str | None:
    # 같은 request/session에서는 같은 seed로 deterministic fake 값을 만들기 위해 seed를 계산한다.
    seed = _get_resource_seed(canonical, world_state)
    # ARN/account field 생성에 사용할 세션 account id를 읽는다.
    account_id = str(
        world_state.get("consistency_locks", {}).get("account_id", "123456789012")
    )
    # ARN/URL field 생성에 사용할 region을 읽는다.
    region = str(world_state.get("region", "us-east-1"))
    # member 이름 비교를 위해 소문자로 정규화한다.
    lowered = member_name.lower()
    # shape 이름과 member 이름을 합쳐 field 의미를 추론한다.
    shape_name = getattr(shape, "name", "")
    combined = f"{shape_name} {member_name}".lower()

    # MonitorInstances 응답 state는 enabled가 자연스럽다.
    if lowered == "state" and canonical.operation == "MonitorInstances":
        return "enabled"
    # UnmonitorInstances 응답 state는 disabled가 자연스럽다.
    if lowered == "state" and canonical.operation == "UnmonitorInstances":
        return "disabled"

    # botocore enum이 있으면 허용 enum 중 하나를 골라 CLI parser 실패를 막는다.
    enum = getattr(shape, "enum", None) or []
    if enum:
        preferred = _pick_enum_value(enum)
        if preferred:
            return preferred

    # Reuse previously seen name-type values, scoped by service to prevent cross-service bleed
    known_names = world_state.get("known_names", {})
    if lowered.endswith("name") or lowered == "name":
        scoped_key = f"{canonical.service}:{member_name}"
        if scoped_key in known_names:
            return known_names[scoped_key]
        # *Name 조기 반환은 구체적 패턴 처리 후 마지막 fallback에서 수행

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
        # Bedrock foundation-model ARN: account 없이 region만 있어야 함
        if canonical.service == "bedrock":
            _known_model_ids = [
                "amazon.titan-text-express-v1",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "meta.llama3-8b-instruct-v1:0",
                "cohere.command-r-v1:0",
            ]
            _idx = int(_det_hex(seed, "bedrock_model_idx", 2), 16) % len(
                _known_model_ids
            )
            return (
                f"arn:aws:bedrock:{region}::foundation-model/{_known_model_ids[_idx]}"
            )
        # IAM ARN: region 없이 account만 있어야 함
        if canonical.service in {"iam", "sts"}:
            return f"arn:aws:{canonical.service}::{account_id}:{canonical.operation.lower()}/{_det_hex(seed, 'arn_suffix', 8)}"
        return f"arn:aws:{canonical.service}:{region}:{account_id}:{canonical.operation.lower()}/{_det_hex(seed, 'arn_suffix', 8)}"
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
            return f"ip-10-42-{_det_int(seed, 'name_a', 10)}-{_det_int(seed, 'name_b', 240) + 10}"
        return f"{canonical.service}-{canonical.operation.lower()}"
    if "digest" in combined:
        return canonical.target_identifiers.get(
            member_name, "sha256:" + _det_hex(seed, "digest", 64)
        )
    if "uploadid" in combined or "upload id" in combined:
        # ECR uploadId는 UUID 형식 (hex only, no "upload-" prefix)
        h = _det_hex(seed, "uploadid", 32)
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    if "jobid" in combined or "job id" in combined:
        return "job-" + _det_hex(seed, "jobid", 10)
    if "requestid" in combined or "request id" in combined:
        return "req-" + _det_hex(seed, "requestid", 16)
    if lowered in ("account", "accountid", "awsaccountid", "ownerid"):
        return account_id
    if lowered.endswith("id"):
        direct = canonical.target_identifiers.get(member_name)
        if direct:
            return direct
        if lowered == "registryid":
            return account_id
        # AvailabilityZoneId: AZ zone ID 형식 (e.g. use1-az1, usw2-az2)
        if lowered == "availabilityzoneid":
            _az_prefix = _region_to_az_prefix(region)
            _az_num = (int(_det_hex(seed, "azid", 2), 16) % 3) + 1
            return f"{_az_prefix}-az{_az_num}"
        exposed_assets = list(world_state.get("exposed_assets", []))
        existing = _find_exposed_asset(lowered, exposed_assets)
        if existing:
            return existing
        if lowered == "instanceid":
            return "i-" + _det_hex(seed, "instanceid", 17)
        if lowered == "reservationid":
            return "r-" + _det_hex(seed, "reservationid", 8)
        if lowered == "reservedinstancesid":
            h = _det_hex(seed, "reservedinstancesid", 32)
            return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
        if lowered == "volumeid":
            return "vol-" + _det_hex(seed, "volumeid", 17)
        if lowered == "imageid":
            return "ami-" + _det_hex(seed, "imageid", 8)
        # Bedrock modelId should look like a real foundation model ID
        if lowered == "modelid" and canonical.service == "bedrock":
            _bedrock_model_ids = [
                "amazon.titan-text-express-v1",
                "amazon.titan-text-lite-v1",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "meta.llama3-8b-instruct-v1:0",
                "cohere.command-r-v1:0",
            ]
            _midx = int(_det_hex(seed, "bedrock_modelid", 2), 16) % len(
                _bedrock_model_ids
            )
            return _bedrock_model_ids[_midx]
        return f"{canonical.service}-{_det_hex(seed, 'generic_id', 8)}"
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
        return "codecommit-" + _det_hex(seed, "credentialalias", 6)
    if lowered == "servicerole":
        return f"arn:aws:iam::{account_id}:role/AWSServiceRoleFor{canonical.service.capitalize()}"
    if "nexttoken" in combined or "marker" in combined or "token" in combined:
        return None
    if "url" in combined:
        return f"mock://{canonical.service}/{canonical.operation.lower()}/{_det_hex(seed, 'url_suffix', 12)}"
    # status message에는 내부 synthetic JSON을 넣지 않고 사람이 볼 수 있는 성공 문구를 넣는다.
    if lowered == "statusmessage":
        return "Operation completed successfully"
    # STS DecodeAuthorizationMessage의 DecodedMessage만 JSON 문자열 의미가 있으므로 별도 처리한다.
    if lowered == "decodedmessage":
        return json.dumps(
            {
                "allowed": False,
                "matchedStatements": [],
                "context": f"synthetic {canonical.service}:{canonical.operation}",
            },
            separators=(",", ":"),
        )
    # 일반 message field에는 synthetic/debug JSON이 새지 않게 평범한 문구를 넣는다.
    if "message" in combined:
        return "Operation completed successfully"
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
        return f"10.42.{_det_int(seed, 'ip_a', 10)}.{_det_int(seed, 'ip_b', 240) + 10}"
    if lowered == "iamrole":
        return "ReadOnlyOpsRole"
    if lowered == "detailedstatus":
        return "Success"
    if lowered == "associationstatus":
        return "Success"
    if lowered == "computername":
        return f"ip-10-42-{_det_int(seed, 'comp_a', 10)}-{_det_int(seed, 'comp_b', 240) + 10}"
    if lowered == "backuprulecron":
        return "cron(0 3 * * ? *)"
    if lowered == "backupruletimezone":
        return "UTC"
    if lowered == "mediatype":
        return "application/vnd.docker.image.rootfs.diff.tar"
    if lowered == "resourcepolicy":
        return '{"Version":"2012-10-17","Statement":[]}'
    if lowered == "reason":
        return "OK"
    if lowered == "lastresourceanalyzed":
        return f"arn:aws:ec2:{region}:{account_id}:instance/i-{_det_hex(seed, 'last_resource', 17)}"
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
            return str(
                world_state.get("session_start_time")
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
    if "availabilityzone" in combined:
        return f"{region}a"
    if "privateip" in combined:
        return (
            f"10.42.{_det_int(seed, 'pip_a', 10)}.{_det_int(seed, 'pip_b', 240) + 10}"
        )
    if "instancetype" in combined:
        return "t3.medium"
    if "state" in combined:
        return "running"
    if "version" in combined:
        return "1"
    # ── Generic patterns to prevent key:key fallback ─────────────────────────
    if any(
        t in combined for t in ("description", "comment", "summary", "note", "detail")
    ):
        return f"{canonical.service.capitalize()} managed resource"
    if "protocol" in combined:
        return "https"
    if "format" in combined:
        return "JSON"
    if "encoding" in combined:
        return "UTF-8"
    if "contenttype" in combined or "content type" in combined:
        return "application/json"
    if any(t in combined for t in ("owner", "creator", "author")):
        return "admin"
    if "path" in combined or "prefix" in combined:
        return "/"
    if lowered.endswith("type") and "instance" not in lowered:
        return "Standard"
    if lowered.endswith("value"):
        direct = canonical.target_identifiers.get(member_name)
        return direct if direct else f"value-{_det_hex(seed, lowered, 8)}"
    if lowered.endswith("key"):
        return f"key-{_det_hex(seed, lowered, 8)}"
    if lowered.endswith("uri"):
        return f"mock://{canonical.service}/{canonical.operation.lower()}/{_det_hex(seed, lowered, 8)}"
    if lowered.endswith("status"):
        return "ACTIVE"
    # *Name 필드: 구체적 패턴이 없으면 이름처럼 보이는 값 생성
    if lowered.endswith("name") and lowered != "name":
        prefix = lowered.removesuffix("name")
        return f"{prefix}-{_det_hex(seed, lowered, 6)}"
    # Last-resort: a short hex string looks more realistic than returning the field name
    return _det_hex(seed, lowered, 12)


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


def _det_hex(seed: str, field: str, length: int) -> str:
    """Derive a deterministic hex string from seed + field tag."""
    raw = hashlib.sha256(f"{seed}:{field}".encode()).hexdigest()
    while len(raw) < length:
        raw += hashlib.sha256(raw.encode()).hexdigest()
    return raw[:length]


def _get_resource_seed(canonical: CanonicalRequest, world_state: dict[str, Any]) -> str:
    """Return a stable seed string tied to the primary resource being queried.

    Incorporates account_id (session-scoped) + resource identifiers (resource-scoped)
    so that: (1) different resources → different seeds, (2) different sessions → different seeds.
    """
    account_id = str(
        world_state.get("consistency_locks", {}).get("account_id", "default")
    )
    # target_identifiers: 정규화된 리소스 ID
    id_values: list[str] = [str(v) for v in canonical.target_identifiers.values() if v]
    # EC2 쿼리 프로토콜 VolumeId.1 / InstanceId.1 형식도 포함
    if not id_values:
        for k, v in canonical.request_params.items():
            if (
                isinstance(v, str)
                and v
                and any(t in k.lower() for t in ("id", "arn", "digest"))
            ):
                id_values.append(v)
    if id_values:
        vals = ":".join(sorted(id_values))
        return f"{account_id}:{vals}"
    # exposed_assets: 이전 응답에서 노출된 instance ID
    for asset in world_state.get("exposed_assets", []):
        if isinstance(asset, str) and re.match(r"^i-[0-9a-f]{8,17}$", asset):
            return f"{account_id}:{asset}"
    # fallback: account_id만으로도 세션 간 격리 보장
    return account_id


def _det_int(seed: str, field: str, max_val: int) -> int:
    """Derive a deterministic integer from a seed + field tag."""
    return int(hashlib.sha256(f"{seed}:{field}".encode()).hexdigest()[:8], 16) % max_val


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


# AWS 리전 → AvailabilityZoneId 접두사 매핑
# 형식: 리전 코드의 두 글자 약어 + 방향/위치 첫 글자 + 숫자 (e.g. us-east-1 → use1)
_REGION_AZ_PREFIX: dict[str, str] = {
    "us-east-1": "use1",
    "us-east-2": "use2",
    "us-west-1": "usw1",
    "us-west-2": "usw2",
    "eu-west-1": "euw1",
    "eu-west-2": "euw2",
    "eu-west-3": "euw3",
    "eu-central-1": "euc1",
    "eu-central-2": "euc2",
    "eu-north-1": "eun1",
    "eu-south-1": "eus1",
    "eu-south-2": "eus2",
    "ap-east-1": "ape1",
    "ap-northeast-1": "apne1",
    "ap-northeast-2": "apne2",
    "ap-northeast-3": "apne3",
    "ap-southeast-1": "apse1",
    "ap-southeast-2": "apse2",
    "ap-southeast-3": "apse3",
    "ap-south-1": "aps1",
    "ap-south-2": "aps2",
    "ca-central-1": "cac1",
    "sa-east-1": "sae1",
    "me-south-1": "mes1",
    "me-central-1": "mec1",
    "af-south-1": "afs1",
}


def _region_to_az_prefix(region: str) -> str:
    """Convert AWS region name to AvailabilityZoneId prefix (e.g. us-east-1 → use1)."""
    if region in _REGION_AZ_PREFIX:
        return _REGION_AZ_PREFIX[region]
    # 알 수 없는 리전은 하이픈 제거 후 앞 4자 사용 (fallback)
    parts = region.split("-")
    if len(parts) >= 3:
        # <geo>-<direction>-<num> 형식
        prefix = parts[0][:2] + parts[1][:1] + parts[-1]
        return prefix[:5]
    return re.sub(r"[^a-z0-9]", "", region)[:4]
