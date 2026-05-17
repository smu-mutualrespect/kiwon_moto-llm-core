# moto-llm-core LLM Callback 흐름 추적 문서

> 가장 빠른 명령어(`ec2 PurchaseReservedInstancesOffering`, avg 995ms)와  
> 가장 느린 명령어(`secretsmanager ValidateResourcePolicy`, avg 2519ms)를  
> 코드 레벨에서 한 줄씩 추적한다.

---

## 목차

1. [시스템 구조 개요](#1-시스템-구조-개요)
2. [공통 진입점: `handle_aws_request()`](#2-공통-진입점-handle_aws_request)
3. [파이프라인 단계별 설명](#3-파이프라인-단계별-설명)
4. [케이스 A: 최단 명령어 (ec2 PurchaseReservedInstancesOffering)](#4-케이스-a-최단-명령어)
5. [케이스 B: 최장 명령어 (secretsmanager ValidateResourcePolicy)](#5-케이스-b-최장-명령어)
6. [두 케이스 비교](#6-두-케이스-비교)
7. [세션 상태 변화 상세](#7-세션-상태-변화-상세)

---

## 1. 시스템 구조 개요

```
AWS CLI / boto3
    │  HTTP 요청 (SigV4 서명)
    ▼
moto_server (Flask HTTP 서버)
    │
    ├─ 네이티브 핸들러가 있는 경우 → Moto 자체 처리 (EC2 RunInstances 등)
    │
    └─ 네이티브 핸들러가 없거나 validate_moto_native_response() == True
           │
           ▼
    handle_aws_request()  ← LLM Callback 진입점
    │   [moto/core/llm_agents/agent.py]
    │
    ├─ 1. extract_session_id_tool()     ← AKIA key → session_id 결정
    ├─ 2. normalize_request_tool()      ← HTTP → CanonicalRequest 변환
    ├─ 3. get_world_state_tool()        ← 세션 상태 로드 (account_id 고정)
    ├─ 4. get_session_history_tool()    ← 이전 요청/응답 이력 텍스트화
    │
    ├─ 5. run_agent_loop()              ← 핵심 루프 (최대 2회 시도)
    │       │  [moto/core/llm_agents/runtime/runner.py]
    │       │
    │       ├─ attempt 1
    │       │   ├─ build_agent_prompt()         ← 시스템 프롬프트 + CURRENT_REQUEST_COMPACT
    │       │   ├─ call_gpt_api_with_meta()     ← OpenAI Responses API 호출
    │       │   ├─ parse_agent_output()         ← LLM JSON 파싱 → AgentOutput
    │       │   ├─ _stabilize_agent_output()    ← tombstone/registry 검증
    │       │   │
    │       │   ├─ [tool_requests 있으면]
    │       │   │   └─ execute_agent_tool_requests()  ← schema/skill/state 툴 실행
    │       │   │       → latest_observation 생성 → attempt 2 진행
    │       │   │
    │       │   └─ [tool_requests 없으면 또는 attempt 2]
    │       │       ├─ build_response_plan_tool()     ← ResponsePlan 생성
    │       │       ├─ adapt_response_plan()          ← field_values 생성
    │       │       │   [moto/core/llm_agents/shape_adapter.py]
    │       │       ├─ serialize_response_tool()      ← XML/JSON 직렬화
    │       │       └─ validate_rendered_response_tool()  ← shape/safety/account 검증
    │       │
    │       └─ AgentRunResult 반환
    │
    ├─ 6. [response_body 비어있으면] _try_xml_fallback() or _ERROR_BODIES
    ├─ 7. add_to_session_history_tool()     ← 이력 기록
    ├─ 8. update_world_state_tool()         ← 세션 상태 업데이트
    ├─ 9. build_comparison_points_tool()    ← 감사용 비교 포인트
    ├─ 10. _write_audit_record()            ← MOTO_LLM_AUDIT_FILE에 JSON 기록
    │
    └─ response_body 반환
```

---

## 2. 공통 진입점: `handle_aws_request()`

**파일**: `moto/core/llm_agents/agent.py:76`

```python
def handle_aws_request(
    service, action, url, headers, body, reason, source
) -> str:
```

이 함수가 LLM fallback의 **유일한 진입점**이다. Moto의 `responses.py`에서 네이티브 핸들러 실패 또는 `validate_moto_native_response()` 판단 후 호출된다.

---

## 3. 파이프라인 단계별 설명

### 3-1. `extract_session_id_tool()` — 세션 ID 결정

**파일**: `moto/core/llm_agents/tools/state_tools.py:63`

```python
auth = headers.get("Authorization")
match = re.search(r"Credential=([A-Z0-9]+)/", auth)
return match.group(1)  # → AKIA 키 문자열
```

SigV4 Authorization 헤더의 `Credential=AKIAXXXXXXXX/...` 에서 access key를 추출한다.  
같은 공격자가 IP를 바꿔도 동일한 session_id를 갖는다. 반대로 다른 공격자가 같은 IP를 써도 session_id가 다르다.

### 3-2. `normalize_request_tool()` → `CanonicalRequest` — HTTP를 내부 표현으로

**파일**: `moto/core/llm_agents/tools/request_tools.py:38`

```python
@dataclass(frozen=True)
class CanonicalRequest:
    service: str          # 예: "ec2", "secretsmanager"
    operation: str        # 예: "PurchaseReservedInstancesOffering"
    principal_type: str   # "iam_user_or_role" (AKIA 감지시)
    probe_style: str      # "enumeration" | "execution"
    raw_action: str       # 원본 action 문자열
    request_params: dict  # body 파싱 결과
    target_identifiers: dict  # id/name/arn 포함 파라미터만 추출
    body_format: str      # "json" | "query" | "xml"
```

- **service 결정 우선순위**: 호출자 명시값 → X-Amz-Target prefix → Authorization credential scope → URL host
- **body_format 결정**: `{`로 시작 → json, `Action=` 포함 → query (EC2/IAM/STS 스타일)
- **target_identifiers**: `arn`, `name`, `id`, `repository`, `secret`, `user` 토큰이 키에 포함된 파라미터만 추출
- **probe_style**: `get/describe/list` 시작 → "enumeration", `assume/run/start` 시작 → "execution", 나머지 → "enumeration"

### 3-3. `get_world_state_tool()` — 세션 상태 초기화 및 로드

**파일**: `moto/core/llm_agents/tools/state_tools.py:81`

세션 최초 진입 시 다음 딕셔너리를 초기화한다:

```python
{
    "session_id": "AKIAXXXXXX",
    "persona": "mid-size-prod-account",
    "region": "us-east-1",
    "phase": "recon",
    "exposed_assets": [],
    "exposed_roles": ["ReadOnlyOpsRole"],
    "credibility_level": "medium",
    "risk_score": 0.2,
    "last_actions": [],
    "consistency_locks": {
        "account_id": _derive_account_id(session_id),  # SHA256 기반 12자리 고정값
        "os_family": "Amazon Linux 2",
    },
    "known_names": {},        # 서비스별 Name 필드 값 캐시
    "agent_responses": [],    # 응답한 operation 목록
    "response_cache": {},     # 동일 read 호출 → 동일 응답 보장
    "resource_registry": {},  # 리소스 ARN/ID/name 매핑 (세션 내 일관성)
    "seen_pagination_tokens": [],
    "session_start_time": "2026-05-17T...",
}
```

**`_derive_account_id(session_id)`**:
```python
digest = int(hashlib.sha256(session_id.encode()).hexdigest()[:10], 16)
return str(100000000000 + (digest % 900000000000))
```
같은 AKIA key면 항상 같은 12자리 account_id가 나온다. 실험에서 `AKIASESSION40CMD` → 결정론적 account_id.

### 3-4. `run_agent_loop()` — 핵심 에이전트 루프

**파일**: `moto/core/llm_agents/runtime/runner.py:27`

```python
for attempt in range(1, max_attempts + 1):  # max_attempts = 2 (기본값)
    agent_output, raw_text, planner_meta = _call_agent_once(...)
    agent_output = _stabilize_agent_output(canonical, world_state, agent_output)
    
    if agent_output.tool_requests and attempt < max_attempts:
        # 툴 실행 후 attempt 2 진행
        latest_observation = execute_agent_tool_requests(...)
        continue
    
    if agent_output.error_mode != "none":
        return 에러 응답
    
    response_plan = build_response_plan_tool(...)
    field_values, plan_meta = adapt_response_plan(...)
    response_body, rendered_meta = serialize_response_tool(...)
    
    if not response_body:
        latest_observation = "serializer returned empty body; ..."
        continue
    
    validation_passed, reason = validate_rendered_response_tool(...)
    if validation_passed:
        return AgentRunResult(...)
    
    latest_observation = f"attempt={attempt} validation_failed reason={reason}; ..."
```

### 3-5. `build_agent_prompt()` — LLM에 보내는 프롬프트

**파일**: `moto/core/llm_agents/runtime/planner.py:61`

프롬프트 구조:

```
{load_agent_system_prompt()}   ← skills/ 디렉토리의 시스템 프롬프트

CURRENT_REQUEST_COMPACT:
svc=ec2
op=PurchaseReservedInstancesOffering
style=enumeration
params={"Action":"PurchaseReservedInstancesOffering","ReservedInstancesOfferingId":"aaaaaa11-...","InstanceCount":"1"}
ids={"ReservedInstancesOfferingId":"aaaaaa11-...","ReservedInstancesOfferingI":"aaaaaa11-..."}
acct=123456789012
region=us-east-1
reason=...
source=...
history="No previous interactions in this session."
tools=["skills.load_skill_document","schema.inspect_output_shape","state.inspect_consistency","mock_data.get_mock_template"]
LATEST_OBSERVATION=None.
OUTPUT_JSON_SCHEMA: {"intent_phase":"recon","response_posture":"sparse|normal","error_mode":"none|...","decoy_bundle_id":"baseline",...}
```

LLM이 반환해야 하는 JSON 스키마:
```json
{
  "intent_phase": "recon",
  "response_posture": "normal",
  "error_mode": "none",
  "decoy_bundle_id": "baseline",
  "risk_delta": 0.1,
  "reason_tags": ["enum_pattern"],
  "tool_requests": [],
  "response_plan": {
    "mode": "success",
    "posture": "normal",
    "entity_hints": {"count": 2},
    "field_hints": {},
    "omit_fields": []
  },
  "environment_delta": {}
}
```

### 3-6. `call_gpt_api_with_meta()` — OpenAI Responses API 호출

**파일**: `moto/core/llm_agents/runtime/provider.py:43`

```python
payload = {
    "model": "gpt-5.4-mini",          # MOTO_LLM_OPENAI_MODEL 환경변수
    "max_output_tokens": 60,           # MOTO_LLM_OPENAI_MAX_OUTPUT_TOKENS (기본 60)
    "reasoning": {"effort": "minimal"},# MOTO_LLM_OPENAI_REASONING_EFFORT
    "input": [{"role": "user", "content": prompt}],
}
```

`POST https://api.openai.com/v1/responses`로 보내고,  
응답의 `output[].content[].text`에서 텍스트를 추출한다.

**비용 구조**: `max_output_tokens=60`으로 출력을 엄격하게 제한. 평균 ~85 output tokens.  
Input tokens는 프롬프트 길이에 따라 700-900 토큰.

### 3-7. `parse_agent_output()` — LLM 응답 파싱

**파일**: `moto/core/llm_agents/runtime/planner.py:95`

```python
parsed = _extract_json(raw_text)  # 마크다운 코드블록 처리 포함
```

- `intent_phase`: {"recon", "privilege_check", "lateral_probe", "impact_probe"} 중 하나
- `response_posture`: {"sparse", "normal", "rich"} 중 하나
- `error_mode`: {"none", "access_denied", "throttling", "not_found"} 중 하나
- `tool_requests`: 최대 3개까지만 허용, dict만
- 유효하지 않은 값은 DEFAULT_OUTPUT 기본값으로 대체

### 3-8. `_stabilize_agent_output()` — 세션 일관성 보정

**파일**: `moto/core/llm_agents/runtime/runner.py:179`

```python
# Phase 1: LLM이 "없다"고 했는데 registry에 있는 경우 → 있는 것으로 보정
if agent_output.error_mode == "not_found":
    if _has_registered_resource_identity(canonical, world_state):
        return replace(agent_output, error_mode="none", response_posture="sparse")

# Phase 2: LLM이 "있다"고 했는데 tombstone 처리된 경우 → 없는 것으로 보정
if agent_output.error_mode == "none" and not _is_delete_operation(canonical):
    if _resource_is_tombstoned(canonical, world_state):
        return replace(agent_output, error_mode="not_found")
```

### 3-9. `execute_agent_tool_requests()` — 툴 실행

**파일**: `moto/core/llm_agents/runtime/tool_executor.py`

지원 툴 4가지:

| 툴 이름 | 역할 |
|---------|------|
| `schema.inspect_output_shape` | botocore output shape 구조 반환 |
| `skills.load_skill_document` | 도메인별 hints 문서 반환 |
| `state.inspect_consistency` | account_id, region, exposed_assets 반환 |
| `mock_data.get_mock_template` | IAM policy 등 템플릿 반환 |

결과는 `TOOL_OBSERVATIONS=[{"tool":"...","output":{...}}]` 형식으로 `latest_observation`에 담겨 attempt 2 프롬프트에 포함된다.

### 3-10. `adapt_response_plan()` — field_values 생성

**파일**: `moto/core/llm_agents/shape_adapter.py:19`

핵심 흐름:
```python
# 1. response_cache 히트 확인
cached = world_state["response_cache"].get(cache_key)
if cached and not is_paginated_continuation:
    return deepcopy(cached), meta  # ← 동일 요청 두 번째는 여기서 즉시 반환

# 2. botocore output shape를 재귀적으로 순회하며 값 생성
payload = _generate_structure(output_shape, ...)
```

**값 생성 우선순위** (각 필드별로 아래 순서로 시도):

1. `resource_registry`에서 이미 기록된 ARN/ID/name 재사용
2. LLM `response_plan.field_hints`에서 명시한 값
3. `request_params` / `target_identifiers`에서 echo
4. `known_names`에서 이전 응답의 Name 재사용
5. botocore enum 목록에서 선호값 선택
6. 필드명/타입 기반 deterministic 생성 (SHA256(seed+field_name))

**seed**: `account_id + ":" + sorted(target_identifier_values)` → 세션+리소스 조합으로 고정

### 3-11. `serialize_response_tool()` — XML/JSON 직렬화

**파일**: `moto/core/llm_agents/tools/render_tools.py`

```python
protocol = service_model.metadata.get("protocol")  # "ec2" | "json" | "query" | ...
serializer_cls = get_serializer_class(service, protocol)
result = serializer_cls(operation_model=operation_model).serialize(field_values)
```

- `ec2`, `query`, `rest-xml` 프로토콜 → XML 생성
- `json`, `rest-json` 프로토콜 → JSON 생성

### 3-12. `validate_rendered_response_tool()` — 검증

**파일**: `moto/core/llm_agents/tools/validation_tools.py:60`

3단계 검증:

```python
# 1. Safety: https://, AKIA 키, private key 패턴 차단 (단 XML namespace의 https:// 허용)
is_safe, _ = _validate_safety(rendered_body)

# 2. Shape: botocore required_members + HONEYPOT_CORE_MEMBERS + 값 형식(InstanceId, ARN 등)
shape_ok, _ = _validate_against_shape(service, operation, rendered_body, check_empty=True)

# 3. World-state: 응답 내 ARN의 account_id가 세션 account_id와 일치하는지
ok, _ = _validate_world_state_consistency(rendered_body, world_state)
```

검증 실패 시 `latest_observation`에 이유를 담아 attempt를 계속한다.

### 3-13. `update_world_state_tool()` — 세션 상태 갱신

**파일**: `moto/core/llm_agents/tools/state_tools.py:117`

응답 완료 후:
- `phase`, `risk_score` 업데이트
- `last_actions`에 `service:operation` 추가 (최근 10개 유지)
- `agent_responses`에 operation 추가
- `environment_delta` 머지
- `exposed_assets`에 ARN/ID 추가
- `known_names`에 Name 필드 값 추가 (서비스 스코프)
- `resource_registry`에 ARN/ID/name 매핑 저장
- **delete 작업이면**: `resource_registry`에 `__deleted__: True` 마크 + `response_cache` 해당 서비스 키 전체 제거
- **read 작업이면**: `response_cache`에 `field_values` 저장 (같은 요청 두 번째부터 캐시 반환)

---

## 4. 케이스 A: 최단 명령어

### 기본 정보

| 항목 | 값 |
|------|-----|
| Command | `aws ec2 purchase-reserved-instances-offering --reserved-instances-offering-id aaaaaa11-bbbb-cccc-ddd-example1 --instance-count 1` |
| Service | ec2 |
| Operation | PurchaseReservedInstancesOffering |
| Protocol | ec2 (XML) |
| Attempt 횟수 | **1** |
| Run1 / Run2 / Run3 | 1001ms / 1000ms / 985ms |
| 평균 | **995ms** |
| Input tokens | 706 |
| Output tokens | 85 |
| Total tokens | 791 |
| 응답 3회 동일 | ✓ |

### Step 1: HTTP Body 파싱

```
body = "Action=PurchaseReservedInstancesOffering&ReservedInstancesOfferingId=aaaaaa11-bbbb-cccc-ddd-example1&InstanceCount=1"
```

`normalize_request_tool` → body_format = **"query"** (EC2 쿼리 프로토콜)

```python
request_params = {
    "Action": "PurchaseReservedInstancesOffering",
    "ReservedInstancesOfferingId": "aaaaaa11-bbbb-cccc-ddd-example1",
    "InstanceCount": "1",
}
target_identifiers = {
    "ReservedInstancesOfferingId": "aaaaaa11-bbbb-cccc-ddd-example1",
    "ReservedInstancesOffering": "aaaaaa11-bbbb-cccc-ddd-example1",  # 복수형 제거 alias
}
```

### Step 2: CanonicalRequest 생성

```python
CanonicalRequest(
    service="ec2",
    operation="PurchaseReservedInstancesOffering",
    principal_type="iam_user_or_role",
    probe_style="enumeration",   # "purchase" → get/describe/list/assume/run/start 아님 → 기본값
    raw_action="PurchaseReservedInstancesOffering",
    request_params={...},
    target_identifiers={"ReservedInstancesOfferingId": "aaaaaa11-..."},
    body_format="query",
)
```

### Step 3: World State 초기화

세션 최초 요청이므로 새 상태 생성:
```python
account_id = _derive_account_id("AKIASESSION40CMD")
# = str(100000000000 + (int(sha256("AKIASESSION40CMD")[:10], 16) % 900000000000))
```

### Step 4: LLM 호출 (1회)

**프롬프트 핵심 부분**:
```
CURRENT_REQUEST_COMPACT:
svc=ec2 op=PurchaseReservedInstancesOffering style=enumeration
params={"Action":"PurchaseReservedInstancesOffering","ReservedInstancesOfferingId":"aaaaaa11-bbbb-cccc-ddd-example1","InstanceCount":"1"}
ids={"ReservedInstancesOfferingId":"aaaaaa11-bbbb-cccc-ddd-example1","ReservedInstancesOffering":"aaaaaa11-bbbb-cccc-ddd-example1"}
acct=<세션_account_id> region=us-east-1
tools=[...4가지...]
LATEST_OBSERVATION=None.
```

**LLM 응답 (추정)**:
```json
{
  "intent_phase": "recon",
  "response_posture": "normal",
  "error_mode": "none",
  "decoy_bundle_id": "baseline",
  "risk_delta": 0.1,
  "reason_tags": ["write_action"],
  "tool_requests": [],
  "response_plan": {
    "mode": "success",
    "posture": "normal",
    "entity_hints": {"count": 1},
    "field_hints": {"reservedInstancesId": "some-uuid"},
    "omit_fields": []
  },
  "environment_delta": {}
}
```

`tool_requests = []` → 툴 실행 없음, attempt 1에서 바로 응답 생성.

### Step 5: `adapt_response_plan()` — 응답 필드 생성

botocore output shape: `PurchaseReservedInstancesOfferingResult`
```
members:
  reservedInstancesId: string
```

`_generate_value("reservedInstancesId", string_shape, ...)`:

1. `_registered_value_for_shape_member` → registry 없음 (첫 요청) → None
2. `_lookup_explicit_hint("reservedInstancesId", ...)` → LLM field_hints에 있으면 사용
3. 없으면 `_generate_scalar_string`:
   - lowered = "reservedinstancesid"
   - UUID 패턴 미충족 → UUID 생성:
     ```python
     h = _det_hex(seed, "reservedinstancesid", 32)
     return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
     ```
   - seed = `account_id + ":" + "aaaaaa11-bbbb-cccc-ddd-example1"`

**결과 field_values**:
```python
{"reservedInstancesId": "a4c763cf-366b-fa94-1d0b-d397110f3d75"}
```

`_normalize_string_hint`에서도 UUID 형식이면 `reservedInstancesId` 재작성 규칙 적용:
```python
# lowered == "reservedinstancesid" and not UUID pattern → UUID 생성
h = _det_hex(value, "reservedinstancesid", 32)
return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
```

### Step 6: XML 직렬화

ec2 프로토콜 → XML serializer:

```xml
<?xml version="1.0" encoding="utf-8"?>
<PurchaseReservedInstancesOfferingResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15">
  <reservedInstancesId>a4c763cf-366b-fa94-1d0b-d397110f3d75</reservedInstancesId>
  <requestId>request-id</requestId>
</PurchaseReservedInstancesOfferingResponse>
```

### Step 7: 검증

- Safety: `https://` 없음 → OK (XML namespace의 amazonaws.com URL은 허용)
- Shape: `<reservedInstancesId>` 존재, XML 파싱 성공 → OK
- World-state: 응답에 12자리 account_id가 포함된 ARN 없음 → OK (reservedInstancesId는 UUID)

검증 통과 → attempt 1에서 완료.

### Step 8: 세션 상태 갱신

```python
# resource_registry에 기록
# target_identifiers의 "ReservedInstancesOfferingId" 사용
registry_keys = [
    "ec2:reservedinstancesofferingid:aaaaaa11-...",
    "ec2:primary:aaaaaa11-...",
]
# 추출된 값: {"reservedInstancesId": "a4c763cf-...", "id": "a4c763cf-..."}
resource_registry["ec2:reservedinstancesofferingid:aaaaaa11-..."] = {
    "reservedInstancesId": "a4c763cf-...",
    "id": "a4c763cf-...",
}

# response_cache 저장? → _should_cache_operation("PurchaseReservedInstancesOffering")
# "purchase"는 get/describe/list/batch/query/head/scan 시작이 아님 → 캐시 안 함
```

### 왜 빠른가?

1. **tool_requests = 0**: LLM이 첫 번째 응답에서 바로 response_plan 생성
2. **botocore output shape 단순**: `reservedInstancesId` 하나
3. **검증 즉시 통과**: UUID 형식, safety 문제 없음
4. **응답이 3회 동일**: seed가 `account_id + offeringId`로 고정 → deterministic UUID → 3번 모두 같은 응답

---

## 5. 케이스 B: 최장 명령어

### 기본 정보

| 항목 | 값 |
|------|-----|
| Command | `aws secretsmanager validate-resource-policy --secret-id prod/db/password --resource-policy '{"Version":"2012-10-17","Statement":[...]}'` |
| Service | secretsmanager |
| Operation | ValidateResourcePolicy |
| Protocol | json |
| Attempt 횟수 | **2** (툴 호출 1회 포함) |
| Run1 / Run2 / Run3 | 2882ms / 2212ms / 2463ms |
| 평균 | **2519ms** |
| Input tokens | ~870 (attempt 1) + retry overhead |
| Output tokens | ~118 |
| Total tokens | ~981 |
| tool_calls_executed | **1** |
| 응답 3회 동일 | ✗ (ValidationErrors 내용 변동) |

### Step 1: HTTP Body 파싱

```json
{
  "SecretId": "prod/db/password",
  "ResourcePolicy": "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":\"*\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"*\"}]}"
}
```

body_format = **"json"**

### Step 2: CanonicalRequest 생성

```python
CanonicalRequest(
    service="secretsmanager",
    operation="ValidateResourcePolicy",
    principal_type="iam_user_or_role",
    probe_style="enumeration",   # "validate" → 열거형으로 분류
    raw_action="ValidateResourcePolicy",
    request_params={
        "SecretId": "prod/db/password",
        "ResourcePolicy": "{...}",
    },
    target_identifiers={
        "SecretId": "prod/db/password",
        "Secret": "prod/db/password",   # 복수형 제거 alias
    },
    body_format="json",
)
```

### Step 3: LLM 호출 (Attempt 1) — 툴 요청 반환

LLM이 `ValidateResourcePolicy` 같은 특수 작업(정책 검증, 불리언 + 에러 목록 반환)에 대해 정확한 응답 구조를 모르기 때문에 schema 확인을 요청한다.

**LLM 응답 (attempt 1 추정)**:
```json
{
  "intent_phase": "recon",
  "response_posture": "normal",
  "error_mode": "none",
  "tool_requests": [
    {"tool": "schema.inspect_output_shape"}
  ],
  "response_plan": {...},
  "environment_delta": {}
}
```

`tool_requests` 비어있지 않음 → `execute_agent_tool_requests()` 호출.

### Step 4: `execute_agent_tool_requests()` — schema 툴 실행

**파일**: `moto/core/llm_agents/runtime/tool_executor.py:51`

```python
if name == "schema.inspect_output_shape":
    return {
        "service": "secretsmanager",
        "operation": "ValidateResourcePolicy",
        "schema": build_full_schema(canonical)[:2000],
    }
```

`build_full_schema`가 botocore에서 `ValidateResourcePolicyResponse` 구조를 읽어 반환:
```json
{
  "service": "secretsmanager",
  "operation": "ValidateResourcePolicy",
  "schema": "output: {PolicyValidationPassed: boolean (required), ValidationErrors: [{CheckName: string, ErrorMessage: string}]}"
}
```

`latest_observation`:
```
TOOL_OBSERVATIONS=[{"tool":"schema.inspect_output_shape","output":{"service":"secretsmanager","operation":"ValidateResourcePolicy","schema":"..."}}]
```

### Step 5: LLM 호출 (Attempt 2) — 실제 응답 생성

두 번째 프롬프트에는 `LATEST_OBSERVATION`에 schema 정보가 포함된다:

```
LATEST_OBSERVATION=TOOL_OBSERVATIONS=[{"tool":"schema.inspect_output_shape","output":{"service":"secretsmanager","operation":"ValidateResourcePolicy","schema":"output: {PolicyValidationPassed: boolean, ValidationErrors: list[{CheckName: string, ErrorMessage: string}]}"}}].
```

**LLM 응답 (attempt 2)**:
```json
{
  "intent_phase": "recon",
  "response_posture": "normal",
  "error_mode": "none",
  "tool_requests": [],
  "response_plan": {
    "mode": "success",
    "posture": "normal",
    "entity_hints": {"count": 1},
    "field_hints": {
      "PolicyValidationPassed": false,
      "ValidationErrors": [{"CheckName": "enum"}]
    },
    "omit_fields": []
  }
}
```

### Step 6: `adapt_response_plan()` — 필드 생성

botocore output shape: `ValidateResourcePolicyResponse`
```
members:
  PolicyValidationPassed: boolean (required → protected member)
  ValidationErrors: list of {
    CheckName: string
    ErrorMessage: string
  }
```

**`PolicyValidationPassed` 생성**:
```python
# _generate_boolean("PolicyValidationPassed", canonical)
lowered = "policyvalidationpassed"
if "validationpassed" in lowered:
    return True   # shape_adapter 기본값
# → 하지만 LLM field_hints에 false가 있으면:
# _lookup_explicit_hint → response_plan.field_hints["PolicyValidationPassed"] = false
# _coerce_explicit_hint(boolean_shape, false) → False
```

**`ValidationErrors` 생성**:
```python
# _generate_list("ValidationErrors", list_shape, ...)
# LLM field_hints에 [{"CheckName": "enum"}]이 있음
# → coerced_items = [_coerce_explicit_hint(item_shape, {"CheckName":"enum"}, ...)]
# → [{"CheckName": "enum"}]
```

**`_requires_non_empty_success("secretsmanager", "ValidateResourcePolicy")`** → True

즉 mode="empty"나 error인 응답이 오면 mode를 "success"로 강제 보정.

**결과 field_values**:
```python
{
    "PolicyValidationPassed": False,
    "ValidationErrors": [{"CheckName": "enum"}]
}
```

### Step 7: JSON 직렬화

```json
{"PolicyValidationPassed": false, "ValidationErrors": [{"CheckName": "enum"}]}
```

### Step 8: 검증

- Safety: https:// 없음 → OK
- Shape:
  - `_HONEYPOT_CORE_MEMBERS[("secretsmanager","ValidateResourcePolicy")]` = `["PolicyValidationPassed"]`
  - `"PolicyValidationPassed"` in payload → OK
  - botocore required: `PolicyValidationPassed` → 존재 → OK
  - value format: boolean → ARN/ID 형식 검사 없음 → OK
- World-state: ARN 포함 없음 → OK

검증 통과 → attempt 2에서 완료.

### 왜 느린가?

| 원인 | 영향 |
|------|------|
| **LLM 2회 호출** | attempt 1 (툴 요청) + attempt 2 (실제 응답) = OpenAI API 왕복 2회 |
| **schema.inspect_output_shape 실행** | botocore 모델 파싱 오버헤드 (소) |
| **응답 구조가 복잡** | boolean + list 중첩 구조, LLM이 스키마 확인 필요 |
| **input tokens 증가** | attempt 2 프롬프트에 TOOL_OBSERVATIONS 포함 → 토큰 증가 |

**OpenAI API 왕복이 2번이므로 실질적으로 2× latency 발생.** Run1 기준:
- attempt 1 LLM 호출: ~600ms
- tool 실행: ~5ms
- attempt 2 LLM 호출: ~600ms
- shape_adapter + serialize + validate: ~100ms
- 기타: ~50ms
- 합계: ~1350ms × (네트워크 변동) = 2882ms

### 3회 응답이 다른 이유

`ValidationErrors` 내의 `ErrorMessage` 필드 등이 LLM non-determinism으로 run마다 달라진다.  
`PolicyValidationPassed=false`는 고정이지만 에러 목록 세부 내용이 변동.

---

## 6. 두 케이스 비교

| 항목 | ec2 Purchase (최단) | secretsmanager Validate (최장) |
|------|--------------------|---------------------------------|
| Protocol | XML (ec2) | JSON |
| body_format | query | json |
| Attempt | 1 | 2 |
| tool_calls | 0 | 1 (schema.inspect_output_shape) |
| botocore output fields | 1개 (reservedInstancesId) | 2개 (boolean + list) |
| 응답 3회 동일 | ✓ (deterministic UUID) | ✗ (ValidationErrors 변동) |
| _requires_non_empty_success | False | **True** (hardcoded) |
| protected_members | 없음 | PolicyValidationPassed |
| response_cache 저장 | ✗ (purchase → write op) | ✗ (validate → 캐시 대상 아님) |
| 평균 latency | **995ms** | **2519ms** |
| 차이 이유 | 단순 shape + attempt 1 완료 | LLM 2회 호출 + 복잡한 구조 |

---

## 7. 세션 상태 변화 상세

### `resource_registry` 에 저장되는 구조

`PurchaseReservedInstancesOffering` 완료 후:
```python
world_state["resource_registry"] = {
    "ec2:reservedinstancesofferingid:aaaaaa11-bbbb-cccc-ddd-example1": {
        "reservedInstancesId": "a4c763cf-366b-fa94-1d0b-d397110f3d75",
        "id": "a4c763cf-366b-fa94-1d0b-d397110f3d75",
    },
    "ec2:primary:aaaaaa11-bbbb-cccc-ddd-example1": {
        "reservedInstancesId": "a4c763cf-...",
        "id": "a4c763cf-...",
    }
}
```

같은 `ReservedInstancesOfferingId`로 다음 요청이 오면 registry에서 찾아 동일한 UUID를 반환한다 → **세션 내 리소스 ID 일관성 보장**.

### `response_cache` 동작

`_should_cache_operation(operation)`:
- `get/describe/list/batch/query/head/scan`으로 시작하는 operation만 캐시
- `Purchase`, `Validate` 등은 캐시 대상 아님

캐시 키 구조:
```python
f"{service}:{operation}:{sha256(target_identifiers)[:8]}"
# 예: "ec2:DescribeVolumeStatus:a1b2c3d4"
```

같은 `DescribeVolumeStatus --volume-ids vol-xxx` 요청이 두 번 오면:
- 첫 번째: LLM 호출 → field_values 생성 → response_cache에 저장
- 두 번째: response_cache에서 즉시 반환 → **LLM 호출 없음** → 수십ms

### Delete 후 상태 변화

예를 들어 ECR `DeleteRepository` 실행 후:
```python
resource_registry["ecr:repositoryname:demo"]["__deleted__"] = True
# + response_cache에서 "ecr:"로 시작하는 모든 키 삭제
```

이후 같은 `demo` repository로 `DescribeRepositories` 요청 시:
1. `_resource_is_tombstoned()` → True
2. `_stabilize_agent_output`: error_mode="none" → "not_found"로 강제
3. `_ERROR_BODIES["not_found"]` 반환

---

## 부록: 각 파일 역할 한 줄 요약

| 파일 | 역할 |
|------|------|
| `agent.py` | LLM callback 유일한 진입점, 전체 파이프라인 조율 |
| `tools/request_tools.py` | HTTP → CanonicalRequest 정규화 |
| `tools/state_tools.py` | 세션 상태 CRUD, resource_registry, response_cache, tombstone |
| `runtime/runner.py` | attempt 루프, _stabilize, AgentRunResult 반환 |
| `runtime/planner.py` | LLM 프롬프트 빌드, 응답 JSON 파싱 |
| `runtime/provider.py` | OpenAI/Anthropic API 실제 호출 |
| `runtime/tool_executor.py` | schema/skill/state/mock_data 툴 실행 |
| `shape_adapter.py` | botocore shape 재귀 순회 → field_values 생성 |
| `tools/planning_tools.py` | ResponsePlan 생성, protected_members, stabilize |
| `tools/render_tools.py` | field_values → XML/JSON 직렬화 |
| `tools/validation_tools.py` | safety/shape/world-state 3단계 검증 |
