# Honeypot Agent Update Report

이 문서는 **커밋끼리 비교한 내용이 아니라**, GitHub `sub_main`에 올라간 최신 커밋을 기준 버전으로 두고 **현재 작업 디렉터리에서 추가로 바뀐 부분**만 정리한 것이다.

비교 기준:

```text
기준 버전: GitHub sub_main 최신 커밋
commit: 08c33d908
message: docs: add .env.example with all supported environment variables

비교 대상: 현재 작업 디렉터리
path: /mnt/c/Users/Administrator/Desktop/kiwon_moto-llm-core-sub_main
```

실제 diff 결과, 허니팟 Agent 관련 변경 파일은 아래 2개다.

```text
moto/core/llm_agents/tools/state_tools.py
tests/test_core/test_llm_agents_runtime.py
```

`validation_tools.py`, `runner.py`, `responses.py`, `provider.py`, `planner.py`, `shape_adapter.py` 등은 GitHub 최신 커밋 대비 현재 디렉터리에서 추가 변경이 없었다.

---

## 1. Agent write 이후 read routing 범위 축소

변경 파일:

```text
moto/core/llm_agents/tools/state_tools.py
```

### 수정 전

GitHub 최신 커밋에서는 Agent가 어떤 service의 write operation을 처리한 적이 있으면, 이후 같은 service의 read operation을 전부 Agent/native-patch 경로로 보낼 수 있었다.

위치:

```text
기준 버전: state_tools.py:31-47
```

수정 전 코드:

```python
def has_cached_agent_response_tool(
    session_id: str, service: str, operation: str
) -> bool:
    """에이전트가 이 operation을 이전에 응답한 적 있는지 확인 (agent_responses 기반).

    같은 세션에서 agent가 해당 서비스의 write 연산을 처리한 적 있으면,
    이후 동일 서비스의 read 연산도 agent로 라우팅해 일관성을 유지한다.
    """
    with _lock:
        state = _session_state.get(session_id, {})
    if f"{service}:{operation}" in state.get("agent_responses", []):
        return True
    # agent가 이 서비스의 write를 처리한 이력이 있고 현재 요청이 read면 agent로 라우팅
    if service in state.get("agent_modified_services", []):
        if _should_cache_operation(operation):
            return True
    return False
```

문제:

```text
ec2:MonitorInstances
-> agent_modified_services = ["ec2"]
-> ec2:DescribeInstances fallback  필요
-> ec2:DescribeImages fallback     불필요
-> ec2:DescribeRegions fallback    불필요
```

즉 `ec2` write 하나 때문에 `ec2` read 전체가 fallback 대상이 되는 구조였다.

### 수정 후

현재 작업 디렉터리에서는 service 단위 broad routing을 제거하고, write operation이 실제로 영향을 줄 수 있는 read operation만 `affected_read_operations`에 기록한다.

위치:

```text
현재 버전: state_tools.py:31-48
```

수정 후 코드:

```python
def has_cached_agent_response_tool(
    session_id: str, service: str, operation: str
) -> bool:
    """에이전트가 이 operation을 이전에 응답한 적 있는지 확인 (agent_responses 기반).

    같은 세션에서 agent가 write 연산을 처리했고 그 write가 현재 read operation에
    영향을 줄 수 있으면 agent/native-patch 경로로 라우팅해 일관성을 유지한다.
    """
    with _lock:
        state = _session_state.get(session_id, {})
    if f"{service}:{operation}" in state.get("agent_responses", []):
        return True
    # 서비스 단위 broad routing은 너무 넓다. write operation이 실제로 영향을 주는
    # read operation만 라우팅한다.
    affected_reads = state.get("affected_read_operations", [])
    if f"{service}:{operation}" in affected_reads:
        return True
    return False
```

변경 의미:

```text
MonitorInstances 이후 DescribeInstances는 fallback 대상
MonitorInstances 이후 DescribeImages는 fallback 대상 아님
```

---

## 2. write operation cache invalidation 위치 수정

변경 파일:

```text
moto/core/llm_agents/tools/state_tools.py
```

### 수정 전

기준 버전에서는 read cache 무효화가 `if field_values:` block 안쪽에 있었다.

위치:

```text
기준 버전: state_tools.py:179-189
```

수정 전 코드:

```python
if field_values:
    known_names = dict(next_state.get("known_names", {}))
    _merge_known_names(known_names, field_values, canonical.service)
    next_state["known_names"] = known_names
    resource_registry = dict(next_state.get("resource_registry", {}))
    _merge_resource_registry(resource_registry, canonical, field_values)
    next_state["resource_registry"] = resource_registry
    if not _should_cache_operation(canonical.operation):
        # delete뿐 아니라 create/update/modify 등 모든 쓰기 연산 후 캐시 무효화
        _invalidate_read_cache(next_state, canonical)
```

문제:

```text
Agent write 응답에서 field_values가 비어 있으면
-> write operation인데도 기존 read cache가 남을 수 있음
-> 이후 read가 stale cache를 반환할 수 있음
```

### 수정 후

현재 버전에서는 write operation이면 `field_values` 유무와 관계없이 먼저 cache를 무효화한다.

위치:

```text
현재 버전: state_tools.py:173-188
```

수정 후 코드:

```python
# agent가 write 연산을 처리했음을 기록 — field_values 유무와 무관하게 항상 실행
if not _should_cache_operation(canonical.operation):
    agent_modified = list(next_state.get("agent_modified_services", []))
    if canonical.service not in agent_modified:
        agent_modified.append(canonical.service)
    next_state["agent_modified_services"] = agent_modified
    _invalidate_read_cache(next_state, canonical)

    affected_reads = list(next_state.get("affected_read_operations", []))
    for read_service, read_operation in _affected_read_operations_for_write(
        canonical
    ):
        read_key = f"{read_service}:{read_operation}"
        if read_key not in affected_reads:
            affected_reads.append(read_key)
    next_state["affected_read_operations"] = affected_reads[-50:]
```

변경 의미:

```text
field_values=None이어도 write operation이면 read cache가 무조건 무효화됨
```

---

## 3. write -> affected read mapping 추가

변경 파일:

```text
moto/core/llm_agents/tools/state_tools.py
```

### 수정 전

기준 버전에는 write operation이 어떤 read operation에 영향을 주는지 나타내는 mapping이 없었다.

위치:

```text
기준 버전: state_tools.py:323 이후 바로 _param_cache_key 진입
```

### 수정 후

현재 버전에는 `_WRITE_TO_AFFECTED_READS`와 `_affected_read_operations_for_write()`가 추가됐다.

위치:

```text
현재 버전: state_tools.py:336-365
```

추가 코드:

```python
_WRITE_TO_AFFECTED_READS: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("ec2", "MonitorInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "UnmonitorInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "RunInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "StartInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "StopInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "TerminateInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "RebootInstances"): (("ec2", "DescribeInstances"),),
    ("ec2", "CreateSecurityGroup"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "AuthorizeSecurityGroupIngress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "AuthorizeSecurityGroupEgress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "RevokeSecurityGroupIngress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "RevokeSecurityGroupEgress"): (("ec2", "DescribeSecurityGroups"),),
    ("ec2", "CreateVolume"): (("ec2", "DescribeVolumes"),),
    ("ec2", "AttachVolume"): (
        ("ec2", "DescribeVolumes"),
        ("ec2", "DescribeInstances"),
    ),
    ("ec2", "DetachVolume"): (
        ("ec2", "DescribeVolumes"),
        ("ec2", "DescribeInstances"),
    ),
    ("ec2", "DeleteVolume"): (("ec2", "DescribeVolumes"),),
}


def _affected_read_operations_for_write(
    canonical: CanonicalRequest,
) -> tuple[tuple[str, str], ...]:
    return _WRITE_TO_AFFECTED_READS.get((canonical.service, canonical.operation), ())
```

변경 의미:

```text
fallback routing 기준이 "service 전체"에서 "operation 영향 범위"로 좁아짐
```

---

## 4. routing/cache 동작 검증 테스트 추가

변경 파일:

```text
tests/test_core/test_llm_agents_runtime.py
```

### 수정 전

기준 버전에는 다음을 직접 검증하는 테스트가 없었다.

```text
write operation 이후 read cache가 무효화되는지
field_values=None이어도 cache가 무효화되는지
affected read만 fallback 대상이 되는지
같은 service의 unrelated read가 fallback 대상에서 제외되는지
```

### 수정 후

현재 버전에 `test_agent_write_invalidates_cache_and_routes_only_affected_reads()`가 추가됐다.

위치:

```text
현재 버전: test_llm_agents_runtime.py:465-495
```

추가 테스트:

```python
def test_agent_write_invalidates_cache_and_routes_only_affected_reads() -> None:
    session_id = "AKIAWRITEAFFECTEDREAD"
    headers = {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={session_id}/20260522/us-east-1/ec2/aws4_request"
        )
    }
    current = get_world_state_tool(session_id, headers)
    current["response_cache"] = {"ec2:DescribeInstances": {"Reservations": []}}
    canonical = normalize_request_tool(
        service="ec2",
        action="MonitorInstances",
        url="https://ec2.us-east-1.amazonaws.com/",
        headers=headers,
        body="Action=MonitorInstances&InstanceId.1=i-1234567890abcdef0",
    )

    update_world_state_tool(
        session_id,
        current,
        canonical,
        DEFAULT_OUTPUT,
        {"assets": []},
        field_values=None,
    )

    updated = get_world_state_tool(session_id, headers)
    assert updated["response_cache"] == {}
    assert has_cached_agent_response_tool(session_id, "ec2", "DescribeInstances")
    assert not has_cached_agent_response_tool(session_id, "ec2", "DescribeImages")
```

검증 의미:

```text
MonitorInstances는 DescribeInstances에만 영향을 준다.
DescribeImages는 같은 ec2 service지만 영향을 받는 read가 아니므로 fallback하지 않는다.
field_values=None이어도 기존 DescribeInstances cache는 제거된다.
```

---

## 5. moto native baseline + resource patch 테스트 추가

변경 파일:

```text
tests/test_core/test_llm_agents_runtime.py
```

### 수정 전

기준 버전에는 `moto_native_body`와 `resource_state_patches`가 실제로 결합되는지 검증하는 테스트가 없었다.

즉 아래 경로가 테스트로 보장되지 않았다.

```text
moto native XML baseline
-> resource_state_patches 적용
-> moto_native_patches provider로 반환
```

### 수정 후

현재 버전에 `test_moto_native_baseline_applies_resource_state_patches()`가 추가됐다.

위치:

```text
현재 버전: test_llm_agents_runtime.py:498-537
```

추가 테스트 핵심:

```python
world_state = {
    "consistency_locks": {"account_id": "123456789012"},
    "resource_state_patches": {
        "ec2": {instance_id: {"Monitoring": {"State": "enabled"}}}
    },
}

result = run_agent_loop(
    canonical=canonical,
    world_state=world_state,
    history_context="",
    reason="unit test",
    source="unit_test",
    moto_native_body=native_body,
)

assert result.planner_meta["provider"] == "moto_native_patches"
assert "<instanceId>i-1234567890abcdef0</instanceId>" in result.response_body
assert "<instanceType>t2.micro</instanceType>" in result.response_body
assert "<state>enabled</state>" in result.response_body
assert "<state>disabled</state>" not in result.response_body
```

검증 의미:

```text
moto native가 만든 정확한 InstanceId/InstanceType은 유지
resource_state_patches의 Monitoring.State=enabled만 덮어씀
기존 disabled 값은 제거
LLM 호출 없이 moto_native_patches 경로로 처리 가능
```

---

## 6. provider 테스트 fake signature 수정

변경 파일:

```text
tests/test_core/test_llm_agents_runtime.py
```

### 수정 전

기준 버전의 provider 테스트 fake function은 `session` keyword argument를 받지 않았다.

위치:

```text
기준 버전: test_llm_agents_runtime.py:824-839
기준 버전: test_llm_agents_runtime.py:869-885
```

수정 전 코드:

```python
def fake_post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout: float,
) -> dict[str, object]:
    ...
```

하지만 provider runtime은 HTTP connection pooling 때문에 `_post_json(..., session=...)` 형태로 호출한다.

기준 버전에서 실제 실행하면 아래 에러가 난다.

```text
TypeError: fake_post_json() got an unexpected keyword argument 'session'
```

### 수정 후

현재 버전에서는 OpenAI/Anthropic 테스트 fake function에 `session` 인자를 추가했다.

위치:

```text
현재 버전: test_llm_agents_runtime.py:905-922
현재 버전: test_llm_agents_runtime.py:952-970
```

수정 후 코드:

```python
def fake_post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout: float,
    session: object,
) -> dict[str, object]:
    captured["url"] = url
    captured["headers"] = headers
    captured["payload"] = payload
    captured["timeout"] = timeout
    captured["session"] = session
    ...
```

변경 의미:

```text
provider._post_json()이 session keyword를 넘겨도 테스트 fake가 TypeError를 내지 않는다.
```

---

## 실제 검증 결과

### 1. 기준 버전에서 provider 테스트 실행

실행 위치:

```text
/tmp/kiwon_moto_remote_sub_main
```

실행:

```bash
MOTO_LLM_ENV_FILE=/mnt/c/Users/Administrator/Desktop/kiwon_moto-llm-core-sub_main/.env pytest \
  tests/test_core/test_llm_agents_runtime.py::test_call_gpt_api_uses_direct_openai_by_default \
  tests/test_core/test_llm_agents_runtime.py::test_call_gpt_api_uses_anthropic_when_only_anthropic_key \
  -q
```

결과:

```text
2 failed
TypeError: fake_post_json() got an unexpected keyword argument 'session'
```

즉 기준 버전 테스트 fake는 현재 provider 호출 방식과 맞지 않았다.

### 2. 기준 버전에서 신규 테스트 존재 여부 확인

기준 버전에는 아래 테스트가 없었다.

```text
test_agent_write_invalidates_cache_and_routes_only_affected_reads
test_moto_native_baseline_applies_resource_state_patches
```

실행 결과:

```text
no tests ran
```

### 3. 현재 작업 디렉터리에서 신규 테스트 실행

실행 위치:

```text
/mnt/c/Users/Administrator/Desktop/kiwon_moto-llm-core-sub_main
```

실행:

```bash
MOTO_LLM_ENV_FILE=.env pytest \
  tests/test_core/test_llm_agents_runtime.py::test_agent_write_invalidates_cache_and_routes_only_affected_reads \
  tests/test_core/test_llm_agents_runtime.py::test_moto_native_baseline_applies_resource_state_patches \
  -q
```

결과:

```text
2 passed in 6.54s
```

### 4. 현재 작업 디렉터리에서 provider 테스트 실행

실행:

```bash
MOTO_LLM_ENV_FILE=.env pytest \
  tests/test_core/test_llm_agents_runtime.py::test_call_gpt_api_uses_direct_openai_by_default \
  tests/test_core/test_llm_agents_runtime.py::test_call_gpt_api_uses_anthropic_when_only_anthropic_key \
  -q
```

결과:

```text
1 passed, 1 failed
```

실패 이유:

```text
assert 1500 == 60
```

해석:

```text
session keyword TypeError는 현재 버전에서 해결됨.
다만 현재 .env의 max output token 설정이 1500으로 로드되어,
OpenAI 테스트가 기대하는 기본값 60과 충돌한다.
```

즉 이번 수정으로 provider fake signature 문제는 해결됐고, 남은 실패는 `.env` 설정값과 테스트 기대값의 충돌이다.

---

## 최종 정리

GitHub `sub_main` 최신 커밋과 현재 작업 디렉터리를 비교했을 때, 현재 로컬에서 추가로 고친 핵심은 다음이다.

1. `state_tools.py`
   - 같은 service 전체를 fallback시키던 broad routing 제거
   - `affected_read_operations` 기반 정밀 routing 추가
   - write operation이면 `field_values=None`이어도 read cache 무효화
   - EC2 write operation별 영향을 받는 read operation mapping 추가

2. `tests/test_core/test_llm_agents_runtime.py`
   - cache invalidation + affected read routing 테스트 추가
   - moto native baseline + resource patch 적용 테스트 추가
   - provider 테스트 fake function에 `session` 인자 추가

3. 검증 결과
   - 기준 버전 provider 테스트는 `session` keyword TypeError로 실패
   - 현재 버전 신규 테스트는 `2 passed in 6.54s`
   - 현재 버전 provider 테스트는 TypeError는 해결됐고, `.env` token 설정값 때문에 OpenAI assertion만 남음
