# Agentic Runtime Cycle Corpus Live 검증 정리

이 문서는 `조회 -> 생성 -> 조회 -> 삭제 -> 조회` cycle corpus를 실제 LLM 호출 모드로 실행했을 때,
LLM 응답이 어떻게 코드 경로를 타고 최종 AWS-like 응답으로 바뀌는지, 그리고 create/read/delete 구간의
repository identity가 왜 동일하게 유지되는지를 정리합니다.

## 결론

최종 검증은 실제 LLM provider를 호출한 `--live` 모드로 수행했습니다.

```bash
python3 scripts/check_agentic_cycle_consistency.py --live \
  --results artifacts/agentic_runtime/ecr_repository_cycle_live_results.json \
  --summary artifacts/agentic_runtime/ecr_repository_cycle_live_summary.md
```

결과:

```text
mode     = live
provider = openai
model    = gpt-5.4-mini
pass     = true
```

검증 파일:

- `artifacts/agentic_runtime/ecr_repository_cycle_corpus.json`
- `artifacts/agentic_runtime/ecr_repository_cycle_live_results.json`
- `artifacts/agentic_runtime/ecr_repository_cycle_live_summary.md`

정정해야 할 점이 있습니다.

AWS lifecycle 관점에서는 1단계 `read_before_create`와 5단계 `read_after_delete`가
create/read/delete identity 동일성 검사의 대상이면 안 됩니다.

- `read_before_create`: 실제 AWS라면 repository가 없으므로 not found가 자연스럽습니다.
- `create`: repository identity가 처음 만들어지는 지점입니다.
- `read_after_create`: create에서 만든 repository와 같은 identity여야 합니다.
- `delete`: 삭제 응답에 포함되는 repository는 create/read_after_create와 같은 identity여야 합니다.
- `read_after_delete`: 실제 AWS라면 not found가 자연스럽습니다.

따라서 이 문서에서 말하는 `Pass: True`는 5단계 전체가 같은 값을 가져도 된다는 뜻이 아니라,
`create`, `read_after_create`, `delete` 세 단계의 identity가 같은지 확인했다는 뜻입니다.

현재 live 결과 파일에는 1단계와 5단계도 repository를 반환한 흔적이 남아 있습니다.
이건 AWS lifecycle 재현으로는 맞지 않고, “sticky fake identity” runtime 성격 때문에 생긴 별도 이슈입니다.
즉 이전 문서의 “5단계 모두 같은 identity” 설명은 잘못된 설명이었습니다.

| 단계 | 명령 | repositoryName | repositoryArn | registryId |
| --- | --- | --- | --- | --- |
| 1 | `aws ecr describe-repositories --repository-names cycle-demo` | `cycle-demo` | `arn:aws:ecr:us-east-1:717494927046:describerepositories/83feb989` | `717494927046` |
| 2 | `aws ecr create-repository --repository-name cycle-demo` | `cycle-demo` | `arn:aws:ecr:us-east-1:717494927046:describerepositories/83feb989` | `717494927046` |
| 3 | `aws ecr describe-repositories --repository-names cycle-demo` | `cycle-demo` | `arn:aws:ecr:us-east-1:717494927046:describerepositories/83feb989` | `717494927046` |
| 4 | `aws ecr delete-repository --repository-name cycle-demo --force` | `cycle-demo` | `arn:aws:ecr:us-east-1:717494927046:describerepositories/83feb989` | `717494927046` |
| 5 | `aws ecr describe-repositories --repository-names cycle-demo` | `cycle-demo` | `arn:aws:ecr:us-east-1:717494927046:describerepositories/83feb989` | `717494927046` |

## 중요한 전제

LLM이 최종 AWS JSON body를 직접 작성하지 않습니다.

실제 흐름은 다음입니다.

1. LLM은 `intent_phase`, `error_mode`, `response_plan` 같은 계획을 냅니다.
2. runtime이 그 계획을 받아 botocore output shape에 맞는 `field_values`를 생성합니다.
3. serializer가 `field_values`를 AWS JSON/XML 응답으로 변환합니다.
4. 세션 state에 `history`, `response_cache`, `resource_registry`를 저장합니다.

따라서 live mode에서는 실제 LLM 호출이 일어나지만, 최종 응답의 ID/ARN/name 안정성은
LLM의 자유 생성이 아니라 runtime의 deterministic state 로직으로 보장됩니다.

## Cycle Corpus

`artifacts/agentic_runtime/ecr_repository_cycle_corpus.json`에는 다음 5개 entry가 있습니다.

```text
1. DescribeRepositories: {"repositoryNames":["cycle-demo"]}
2. CreateRepository:     {"repositoryName":"cycle-demo"}
3. DescribeRepositories: {"repositoryNames":["cycle-demo"]}
4. DeleteRepository:     {"repositoryName":"cycle-demo","force":true}
5. DescribeRepositories: {"repositoryNames":["cycle-demo"]}
```

runner는 모든 요청에 같은 fake SigV4 credential을 넣습니다.

```text
Credential=AKIACYCLECORPUS/20260517/us-east-1/ecr/aws4_request
```

이 access key 부분인 `AKIACYCLECORPUS`가 session id가 됩니다. 그래서 5개 요청은 같은 session state를 공유합니다.

## Entry Point: `handle_aws_request`

LLM fallback의 진입점은 `moto/core/llm_agents/agent.py`의 `handle_aws_request()`입니다.

`agent.py:88-91`:

```python
session_id = extract_session_id_tool(headers)
canonical = normalize_request_tool(service, action, url, headers, body)
world_state = get_world_state_tool(session_id, headers)
history_context = get_session_history_tool(session_id)
```

여기서 결정되는 값:

```python
session_id = "AKIACYCLECORPUS"
canonical.service = "ecr"
canonical.operation = "DescribeRepositories" | "CreateRepository" | "DeleteRepository"
canonical.target_identifiers = {"repositoryName": "cycle-demo", ...}
```

`agent.py:93-100`에서 `run_agent_loop()`를 호출합니다.

```python
run_result = run_agent_loop(
    canonical=canonical,
    world_state=world_state,
    history_context=history_context,
    reason=reason,
    source=source,
    max_attempts=_max_attempts(),
)
```

응답이 만들어진 뒤에는 `agent.py:115-122`에서 history와 world state가 저장됩니다.

```python
add_to_session_history_tool(...)
update_world_state_tool(...)
```

## Session State

session id는 `moto/core/llm_agents/tools/state_tools.py`에서 뽑습니다.

`state_tools.py:63-71`:

```python
auth = str(headers.get("Authorization") or headers.get("authorization") or "")
if auth:
    match = re.search(r"Credential=([A-Z0-9]+)/", auth)
    if match:
        return match.group(1)
```

처음 보는 session이면 `state_tools.py:81-106`에서 world state를 만듭니다.

중요 필드:

```python
"response_cache": {},
"resource_registry": {},
"consistency_locks": {
    "account_id": _derive_account_id(session_id),
    "os_family": "Amazon Linux 2",
}
```

이번 session의 account id는 deterministic하게 `717494927046`으로 잡혔고,
그래서 모든 `registryId`와 ARN account id가 이 값으로 유지됩니다.

## Live LLM 호출 위치

`moto/core/llm_agents/runtime/runner.py`의 `_call_agent_once()`가 실제 provider 호출 지점입니다.

`runtime/runner.py:154-165`:

```python
prompt = build_agent_prompt(...)
raw, meta = call_gpt_api_with_meta(prompt)
```

`--live` 모드에서는 여기서 실제 LLM 호출이 일어납니다. 이번 실행 로그에서도 각 단계가 다음처럼 찍혔습니다.

```text
provider : openai (gpt-5.4-mini)
```

반대로 offline mode에서는 `runtime/runner.py:145-152`에서 `DEFAULT_OUTPUT`을 반환합니다.

```python
return (
    DEFAULT_OUTPUT,
    "",
    {
        "provider": "offline_stub",
        "model": "deterministic_response_plan",
        ...
    },
)
```

즉 이번 최종 검증은 offline stub이 아니라 `call_gpt_api_with_meta()`를 탄 실제 live 호출입니다.

## LLM 응답 이후 흐름

`runtime/runner.py:39-49`:

```python
agent_output, raw_text, planner_meta = _call_agent_once(...)
agent_output = _stabilize_agent_output(canonical, world_state, agent_output)
```

그 다음 `runtime/runner.py:82-88`:

```python
response_plan = build_response_plan_tool(
    canonical, agent_output, world_state, raw_text
)
field_values, plan_meta = adapt_response_plan(
    canonical, response_plan, world_state
)
response_body, rendered_meta = serialize_response_tool(canonical, field_values)
```

이 단계별 역할:

| 단계 | 역할 |
| --- | --- |
| `_call_agent_once()` | 실제 LLM 호출 또는 offline stub 반환 |
| `_stabilize_agent_output()` | LLM이 낸 error 판단을 세션 state 기준으로 안정화 |
| `build_response_plan_tool()` | LLM raw JSON을 `ResponsePlan`으로 정규화 |
| `adapt_response_plan()` | botocore output shape에 맞는 Python dict 생성 |
| `serialize_response_tool()` | AWS protocol JSON/XML body로 직렬화 |
| `validate_rendered_response_tool()` | shape/safety/world-state 검증 |

검증은 `runtime/runner.py:94-105`에서 수행합니다.

```python
validation_passed, validation_reason = validate_rendered_response_tool(...)
if validation_passed:
    return AgentRunResult(...)
```

## Identity가 유지되는 핵심 1: Read Cache

`DescribeRepositories`처럼 read operation은 같은 요청에 대해 같은 payload를 반환해야 합니다.

`shape_adapter.py:38-52`:

```python
cached = world_state.get("response_cache", {}).get(
    _param_cache_key(
        canonical.service, canonical.operation, canonical.target_identifiers
    )
)
if cached is not None and not is_paginated_continuation:
    payload = deepcopy(cached)
    return payload, meta
```

그래서 1단계 `DescribeRepositories` 결과가 cache에 저장되면,
3단계와 5단계의 동일한 describe 요청은 새 ARN을 만들지 않고 cached payload를 그대로 반환합니다.

cache 저장은 `state_tools.py:171-179`입니다.

```python
if field_values and _should_cache_operation(canonical.operation):
    response_cache[cache_key] = deepcopy(field_values)
```

## Identity가 유지되는 핵심 2: Resource Registry

create/delete는 read cache를 직접 쓰지 않습니다. 대신 logical resource identity를 `resource_registry`에 저장하고 재사용합니다.

저장은 `state_tools.py:162-169`에서 시작합니다.

```python
resource_registry = dict(next_state.get("resource_registry", {}))
_merge_resource_registry(resource_registry, canonical, field_values)
next_state["resource_registry"] = resource_registry
```

실제 추출은 `state_tools.py:325-355`입니다.

```python
if lowered.endswith("arn") and value.startswith("arn:aws:"):
    extracted.setdefault(key, value)
    extracted.setdefault("arn", value)
elif lowered.endswith("id") and _looks_like_resource_id(value):
    extracted.setdefault(key, value)
    extracted.setdefault("id", value)
elif (lowered == "name" or lowered.endswith("name")) and value:
    extracted.setdefault(key, value)
    extracted.setdefault("name", value)
```

`repositoryName=cycle-demo`이면 registry key는 이런 식으로 잡힙니다.

```text
ecr:repositoryname:cycle-demo
ecr:primary:cycle-demo
```

그래서 operation이 `DescribeRepositories`, `CreateRepository`, `DeleteRepository`로 달라도
같은 `repositoryName=cycle-demo`이면 같은 registry 값을 찾습니다.

조회는 `shape_adapter.py:999-1035`입니다.

```python
if lowered.endswith("arn") or lowered == "arn":
    return _lookup_registered_resource_value(..., "arn")
if lowered.endswith("id"):
    return _lookup_registered_resource_value(..., "id")
if lowered == "name" or lowered.endswith("name"):
    return _lookup_registered_resource_value(..., "name")
```

그리고 string field 생성 초반에 registry를 먼저 봅니다.

`shape_adapter.py:129-136`:

```python
if shape.type_name == "string":
    registered = _registered_value_for_shape_member(
        canonical, world_state, member_name
    )
    if registered:
        return _normalize_string_hint(
            registered, canonical, world_state, member_name
        )
```

즉 LLM이 다른 값을 hint로 줘도, 이미 같은 logical resource가 registry에 있으면 registry 값이 우선됩니다.

## Identity가 유지되는 핵심 3: Protected Members

ECR repository 응답에서 반드시 안정화해야 하는 필드는 protected member로 지정했습니다.

`shape_adapter.py:1061-1076`:

```python
ecr_repository_members = {
    "repository",
    "repositories",
    "repositoryArn",
    "registryId",
    "repositoryName",
}
if operation_key == ("ecr", "DescribeRepositories"):
    protected.update(ecr_repository_members)
elif operation_key == ("ecr", "CreateRepository"):
    protected.update(ecr_repository_members)
elif operation_key == ("ecr", "DeleteRepository"):
    protected.update(ecr_repository_members)
```

이번 live 검증 중 실제로 LLM이 `DeleteRepository`에서 다음처럼 빈 구조 hint를 준 케이스가 있었습니다.

```json
{
  "response_plan": {
    "field_hints": {
      "repository": {}
    }
  }
}
```

기존에는 이 빈 dict를 그대로 신뢰해서 삭제 응답의 `repository`가 비어 버렸습니다.

현재는 `shape_adapter.py:138-146`에서 explicit hint를 적용하기 전에 protected empty hint인지 검사합니다.

```python
if (
    explicit is not None
    and not _protected_empty_hint(member_name, shape, explicit, protected_members)
    and _explicit_hint_is_compatible(shape, explicit)
):
    return _coerce_explicit_hint(...)
```

`shape_adapter.py:287-300`:

```python
def _protected_empty_hint(...):
    if member_name not in protected_members:
        return False
    if type_name == "structure":
        return isinstance(explicit, dict) and not explicit
    if type_name == "list":
        return isinstance(explicit, list) and not explicit
    return False
```

그래서 protected field인 `repository`에 `{}`가 들어오면 LLM hint를 무시하고 botocore shape 기반 생성으로 넘어갑니다.

## Identity가 유지되는 핵심 4: not_found 안정화

이번 live 검증 중 또 하나의 실제 실패는 LLM이 삭제 단계에서 `error_mode=not_found`를 낸 것입니다.

기존 흐름에서는 `runtime/runner.py:67-80`에 의해 error body로 빠졌습니다.

```python
if agent_output.error_mode != "none":
    return AgentRunResult(
        response_body="",
        field_values={},
        ...
    )
```

그 결과 `agent.py:107-113`에서 fallback error body가 만들어졌습니다.

```python
response_body = _try_xml_fallback(canonical) or _ERROR_BODIES.get(
    agent_output.error_mode, _ERROR_BODIES["access_denied"]
)(canonical.service, canonical.operation)
```

삭제 단계 결과가 `ResourceNotFoundException`이 되면 `repositoryName`, `repositoryArn`, `registryId`가 빠지므로 cycle consistency가 깨집니다.

수정 후에는 `runtime/runner.py:178-187`에서 같은 session registry에 이미 동일 resource가 있는지 확인합니다.

```python
def _stabilize_agent_output(...):
    if agent_output.error_mode != "not_found":
        return agent_output
    if not _has_registered_resource_identity(canonical, world_state):
        return agent_output
    return replace(agent_output, error_mode="none", response_posture="sparse")
```

즉 이미 이 session에서 `cycle-demo` repository를 보여준 적이 있으면,
LLM의 `not_found` 판단을 그대로 에러로 내보내지 않고 success response path로 안정화합니다.

## 5단계별 실제 처리

### 1단계: 조회 before create

명령:

```bash
aws ecr describe-repositories --repository-names cycle-demo
```

처리:

1. AWS lifecycle 기준으로는 여기서 repository가 없어야 합니다.
2. 따라서 이상적인 응답은 `RepositoryNotFoundException` 계열입니다.
3. 이 단계는 create/read/delete identity 동일성 검사 대상이 아닙니다.

현재 live 결과에서 이 단계가 repository를 반환한 것은 lifecycle 관점에서는 맞지 않습니다.
이 값은 뒤 단계 identity 검증의 기준으로 쓰면 안 됩니다.

정리하면, 이 단계는 “생성 전 조회가 not found처럼 처리되는가”를 따로 봐야 하는 lifecycle 검사 항목입니다.

### 2단계: 생성

명령:

```bash
aws ecr create-repository --repository-name cycle-demo
```

처리:

1. create는 read operation이 아니라 `DescribeRepositories` cache를 직접 쓰지 않습니다.
2. `repository` 구조를 생성합니다.
3. string field 생성 시 `resource_registry`를 먼저 봅니다.
4. 기존 registry 값이 없으면 여기서 `cycle-demo`의 identity가 만들어집니다.
5. 응답 후 `resource_registry`에 `repositoryArn`, `registryId`, `repositoryName`이 저장됩니다.

이 단계가 create/read/delete identity 비교의 기준입니다.

### 3단계: 생성 후 조회

명령:

```bash
aws ecr describe-repositories --repository-names cycle-demo
```

처리:

1. 생성 이후 조회이므로 create 단계에서 만든 repository와 같은 logical resource를 찾아야 합니다.
2. `resource_registry`에 저장된 `cycle-demo` identity를 재사용합니다.
3. 이 단계의 `repositoryArn`, `registryId`, `repositoryName`은 create 단계와 같아야 합니다.

결과는 2단계 create와 같아야 합니다.

### 4단계: 삭제

명령:

```bash
aws ecr delete-repository --repository-name cycle-demo --force
```

처리:

1. 실제 LLM 호출이 발생합니다.
2. LLM이 `not_found`를 내더라도 registry에 `cycle-demo`가 있으면 success path로 안정화합니다.
3. LLM이 `repository: {}`를 내더라도 protected empty hint라서 무시합니다.
4. `DeleteRepository` output shape의 `repository` 구조를 생성합니다.
5. `repositoryArn`, `registryId`, `repositoryName`은 registry에서 재사용합니다.

결과는 2단계 create, 3단계 read_after_create와 같아야 합니다.

### 5단계: 삭제 후 조회

명령:

```bash
aws ecr describe-repositories --repository-names cycle-demo
```

처리:

1. AWS lifecycle 기준으로는 delete 이후 repository가 없어야 합니다.
2. 따라서 이상적인 응답은 `RepositoryNotFoundException` 계열입니다.
3. 이 단계는 create/read/delete identity 동일성 검사 대상이 아닙니다.

현재 live 결과에서 이 단계가 cached repository를 반환한 것은 lifecycle 관점에서는 맞지 않습니다.
즉 별도 lifecycle check를 추가한다면 이 부분은 아직 수정 대상입니다.

## 실제 Live 검증에서 잡힌 문제

이번 live 검증에서 실제로 실패했던 케이스는 두 개였습니다.

첫 번째 실패:

```text
DeleteRepository response_body = {"repository": {}}
```

원인:

- LLM이 `field_hints.repository = {}`를 냈습니다.
- 기존 adapter가 빈 구조 hint를 그대로 신뢰했습니다.
- 그래서 삭제 응답에서 `repositoryName`, `repositoryArn`, `registryId`가 빠졌습니다.

수정:

- `shape_adapter.py:138-146`
- `shape_adapter.py:287-300`

protected member의 빈 dict/list hint는 무시하도록 했습니다.

두 번째 실패:

```text
DeleteRepository response_body = {"__type":"ResourceNotFoundException", ...}
```

원인:

- LLM이 삭제 단계에서 `error_mode=not_found`를 냈습니다.
- 기존 runner가 error mode를 그대로 에러 응답으로 반환했습니다.
- cycle consistency 검사에서 삭제 단계 repository identity가 비었습니다.

수정:

- `runtime/runner.py:49`
- `runtime/runner.py:178-187`

같은 session의 `resource_registry`에 이미 logical resource가 있으면 `not_found`를 success path로 안정화했습니다.

## 검증 결과

단위 테스트:

```bash
pytest tests/test_core/test_llm_agents_runtime.py
```

결과:

```text
24 passed
```

Live cycle:

```bash
python3 scripts/check_agentic_cycle_consistency.py --live \
  --results artifacts/agentic_runtime/ecr_repository_cycle_live_results.json \
  --summary artifacts/agentic_runtime/ecr_repository_cycle_live_summary.md
```

결과:

```text
Pass: True
```

이 `Pass: True`는 create/read_after_create/delete 세 단계의 identity 동일성 기준입니다.
`read_before_create`와 `read_after_delete`의 AWS lifecycle 정확성까지 통과했다는 뜻은 아닙니다.

## AWS 실제 동작과의 차이

이번 검증은 실제 AWS CLI를 AWS 계정에 붙여서 호출한 검증이 아닙니다.

검증한 것은 이것입니다.

```text
실제 LLM provider를 호출했을 때,
agentic runtime이 같은 세션의 fake AWS resource identity를 안정적으로 유지하는가
```

실제 AWS라면 생성 전 describe와 삭제 후 describe는 보통 not found가 됩니다. 현재 runtime은 그 lifecycle 의미를
완전히 반영하지 않고, 이미 노출한 fake resource identity를 세션 안에서 계속 유지하는 쪽으로 동작합니다.

따라서 이 cycle corpus의 pass 기준은 “AWS 실제 상태와 동일한가”가 아니라
“LLM 호출이 섞여도 create/read/delete 구간의 identity가 바뀌지 않는가”입니다.

AWS lifecycle까지 맞추려면 추가 수정이 필요합니다.

- 생성 전 조회는 not found로 반환
- 생성 후 조회는 create identity 반환
- 삭제 응답은 create identity 반환
- 삭제 후 조회는 not found로 반환
