# LLM Fallback Diff Fix Report

## 목적

이 문서는 `sub_main` 브랜치에서 LLM fallback 관련 오류를 고치기 위해 어떤 diff가 들어갔고, 각 diff가 어떤 오류를 해결했는지 정리한다.

검증 기준은 사용자가 제공한 AWS CLI 40개 명령이다.

## 최종 결과

수정 전 server-mode 결과:

- 성공: `23`
- 실패: `18`
- 비교용 `healthomics`/`omics` 포함 시 41 row
- 주요 실패 유형:
  - Flask `500 Internal Server Error`
  - `NoneType` backend crash
  - native Moto empty-lab error 노출
  - AWS CLI union parse error
  - AWS-like 하지 않은 ID/message 값
  - `healthomics` invalid command
  - `omics` host-prefix endpoint rewrite

수정 후 server-mode 결과:

- 유효한 40개 명령: `40 OK`
- 실패: `0`
- fallback audit records: `40`
- server traceback: `0`
- targeted runtime test: `pytest -q tests/test_core/test_llm_agents_runtime.py` -> `20 passed`

## 1. Server Backend 추론 실패 시 Flask 500이 나던 문제

### 기존 증상

다음 계열 명령에서 native Moto backend를 찾지 못하고 Flask HTML 500이 반환됐다.

- `resource-explorer-2 list-indexes`
- `resource-explorer-2 list-views`
- `resource-explorer-2 search`
- `accessanalyzer list-analyzers`
- `billingconductor list-billing-groups`
- `frauddetector get-detectors`
- `backup-gateway list-gateways`

대표 로그:

```text
AttributeError: 'NoneType' object has no attribute 'replace'
```

### 원인

`moto/moto_server/werkzeug_app.py`의 `get_application()`이 request host/path/body에서 backend를 못 찾으면 `service=None` 상태가 될 수 있었다.

그 상태로 native app 생성을 계속하면 내부에서 `backends.get_backend(None)` 흐름으로 들어가고, backend 이름을 문자열로 가정하는 코드에서 crash가 발생했다.

### 바뀐 파일

- `moto/moto_server/werkzeug_app.py`

### diff 요지

추가된 service/action 추론 로직:

- SigV4 `Authorization` credential scope에서 `region/service` 추출
- `X-Amz-Target` prefix에서 service 추출
- `X-Amz-Target` suffix에서 operation 추출
- Query protocol body의 `Action=`에서 operation 추출
- REST path와 botocore service model을 비교해 operation 추출

추가된 alias:

```python
SIGNING_ALIASES = {
    "access-analyzer": "accessanalyzer",
    ...
}

TARGET_PREFIX_ALIASES = {
    "AWSHawksNestServiceFacade": "frauddetector",
    "AWSResourceExplorer": "resource-explorer-2",
    "BackupOnPremises_v20210101": "backup-gateway",
}
```

핵심 변경:

```python
if not backend:
    app = self.app_instances.get("__llm_fallback__", None)
    if app is None:
        app = create_llm_fallback_app()
        self.app_instances["__llm_fallback__"] = app
    return app
```

즉, backend를 끝까지 못 찾으면 더 이상 native backend app을 만들지 않고 LLM fallback app으로 보낸다.

### 해결 결과

기존에 Flask HTML 500을 내던 unknown backend 계열 요청이 `handle_aws_request()`로 들어가 AWS-shaped JSON/XML 응답을 반환하게 됐다.

예시:

```bash
aws --endpoint-url=http://127.0.0.1:<port> resource-explorer-2 list-indexes
```

수정 후:

```json
{
  "Indexes": [
    {
      "Region": "us-east-1",
      "Arn": "arn:aws:resource-explorer-2:us-east-1:<account>:listindexes/...",
      "Type": "LOCAL"
    }
  ]
}
```

## 2. Fallback Runtime이 service를 `unknown`으로 잡던 문제

### 기존 증상

server fallback app까지 요청이 들어와도, runtime normalizer가 service를 복구하지 못하면 `unknown:Operation`처럼 agent context가 부정확해질 수 있었다.

그 결과 botocore output shape 조회, protocol 선택, response rendering 정확도가 떨어진다.

### 원인

`moto/core/llm_agents/tools/request_tools.py`의 `normalize_request_tool()`은 기존에 주로 URL host 기반으로 service를 추론했다.

하지만 server-mode endpoint URL은 보통 다음처럼 로컬 host다.

```text
http://127.0.0.1:<port>/...
```

이 경우 host만 봐서는 AWS service를 알 수 없다.

### 바뀐 파일

- `moto/core/llm_agents/tools/request_tools.py`

### diff 요지

service 추론 순서가 확장됐다.

기존:

```python
normalized_service = (service or _service_from_host(url) or "unknown").lower()
```

수정 후:

```python
normalized_service = (
    service
    or _service_from_target(headers)
    or _service_from_authorization(headers)
    or _service_from_host(url)
    or "unknown"
).lower()
```

추가된 복구 경로:

- `_service_from_target(headers)`
- `_service_from_authorization(headers)`

### 해결 결과

로컬 endpoint URL이어도 다음 정보에서 service를 복구한다.

- `Authorization: Credential=.../us-east-1/resource-explorer-2/aws4_request`
- `X-Amz-Target: AWSResourceExplorer.ListIndexes`

그 결과 audit의 canonical request가 정확해졌다.

예시:

```json
{
  "service": "resource-explorer-2",
  "operation": "ListIndexes",
  "principal_type": "iam_user_or_role",
  "probe_style": "enumeration"
}
```

## 3. Native Moto empty-lab error가 그대로 노출되던 문제

### 기존 증상

일부 명령은 native Moto handler까지 도달했지만, 허니팟 관점에서 부적절한 빈 환경 에러를 그대로 반환했다.

예시:

```text
Stack with id prod-app does not exist
```

```text
Your account is not a member of an organization.
```

### 원인

기존 fallback 조건은 다음 케이스를 주로 처리했다.

- handler 미구현
- action 미확인
- native response shape mismatch
- 같은 세션에서 agent가 이미 만든 리소스 ID의 후속 not-found

하지만 `cloudformation describe-stack-resources`, `organizations list-roots`처럼 native error 자체가 빈 실험 환경을 노출하는 경우는 fallback 대상이 아니었다.

### 바뀐 파일

- `moto/core/responses.py`

### diff 요지

native error fallback 대상 operation이 추가됐다.

```python
_HONEYPOT_NATIVE_ERROR_OPERATIONS = {
    ("cloudformation", "DescribeStackResources"),
    ("organizations", "ListRoots"),
}
```

native error body matcher가 추가됐다.

```python
_HONEYPOT_NATIVE_ERROR_RE = re.compile(
    r"ValidationError|NotFound|NoSuchEntity|NoSuchKey|does\s+not\s+exist"
    r"|AWSOrganizationsNotInUseException|not\s+a\s+member\s+of\s+an\s+organization",
    re.IGNORECASE,
)
```

`BaseResponse.call_action()`의 fallback 판단에 다음 조건이 추가됐다.

```python
or _native_error_should_fallback_for_honeypot(
    self.service_name, self._get_action(), status, body
)
```

### 해결 결과

빈 lab을 드러내는 native error가 공격자에게 바로 노출되지 않고, agent가 plausible AWS 응답을 생성한다.

예시:

```bash
aws cloudformation describe-stack-resources --stack-name prod-app
aws organizations list-roots
```

수정 후 두 명령 모두 `OK`가 됐다.

## 4. Native 성공 응답이 빈 inventory를 노출하던 문제

### 기존 증상

일부 list 계열 명령은 native Moto가 성공 응답을 반환하지만, 허니팟 관점에서는 비어 있는 계정처럼 보일 수 있었다.

예:

- `cloudformation list-stacks`
- `organizations list-accounts`
- `backup list-backup-vaults`

### 원인

native 성공 응답은 shape가 맞으면 그대로 통과했다.

하지만 허니팟은 “정상적인 빈 계정”보다 “그럴듯한 decoy surface”가 더 중요하다.

### 바뀐 파일

- `moto/core/responses.py`

### diff 요지

성공 응답이어도 fallback을 강제할 recon operation 목록이 추가됐다.

```python
_HONEYPOT_FORCE_RECON_FALLBACK_OPERATIONS = {
    ("backup", "ListBackupVaults"),
    ("cloudformation", "ListStacks"),
    ("organizations", "ListAccounts"),
}
```

`BaseResponse.call_action()` fallback 판단에 다음 조건이 추가됐다.

```python
or _native_success_should_fallback_for_honeypot(
    self.service_name, self._get_action(), status
)
```

환경 변수로 끌 수 있다.

```text
MOTO_LLM_HONEYPOT_FORCE_RECON_FALLBACK=0
```

### 해결 결과

고가치 recon 명령이 native empty inventory 대신 LLM fallback으로 들어가고, audit에도 기록된다.

최종 40개 valid command 실행에서 fallback audit records가 `40`개로 맞춰졌다.

## 5. AWS CLI가 union shape 응답을 파싱하지 못하던 문제

### 기존 증상

`accessanalyzer list-analyzers` 응답에서 botocore union shape가 여러 member를 동시에 포함하면 AWS CLI parser가 실패할 수 있었다.

### 원인

`shape_adapter.py`의 structure generator가 모든 member를 순회하면서 값을 만들었다.

union shape는 AWS 규칙상 여러 member 중 하나만 존재해야 한다.

### 바뀐 파일

- `moto/core/llm_agents/shape_adapter.py`

### diff 요지

structure 생성 시 union metadata를 확인하고 첫 member만 생성하도록 변경했다.

```python
members = list(shape.members.items())
if getattr(shape, "metadata", {}).get("union"):
    members = members[:1]
for member_name, member_shape in members:
    ...
```

### 해결 결과

`accessanalyzer list-analyzers`가 AWS CLI parse error 없이 `OK`가 됐다.

추가 테스트:

```python
test_shape_adapter_generates_single_member_for_union_shapes
```

## 6. 요청에 포함된 AWS ID가 응답에서 깨지던 문제

### 기존 증상

EC2 계열 응답에서 요청한 instance id가 보존되지 않거나, AWS-like 하지 않은 값이 나왔다.

예:

```text
instanceid-12345abcde
```

AWS CLI/허니팟 관점에서 자연스러운 값:

```text
i-1234567890abcdef0
```

### 원인

Query protocol은 list parameter를 `InstanceId.1`처럼 전달한다.

기존 `_lookup_explicit_hint()`는 `InstanceId.1` 변형을 찾지 못해 요청 값을 놓쳤고, generic string generator로 빠질 수 있었다.

### 바뀐 파일

- `moto/core/llm_agents/shape_adapter.py`

### diff 요지

명시 hint 후보에 `.1` 변형을 추가했다.

```python
candidates = [
    member_name,
    f"{member_name}.1",
    member_name[:1].lower() + member_name[1:],
    f"{member_name[:1].lower() + member_name[1:]}.1",
    member_name.lower(),
    f"{member_name.lower()}.1",
]
```

잘못된 AWS ID hint를 AWS-like 형식으로 보정하는 로직이 추가됐다.

```python
if lowered == "instanceid" and not re.match(r"^i-[0-9a-f]{8,17}$", value):
    return "i-" + _det_hex(value, "instanceid", 17)
```

비슷한 보정이 다음 ID에도 추가됐다.

- `VolumeId`
- `VpcId`
- `SubnetId`
- `GroupId`
- `SecurityGroupId`

### 해결 결과

`ec2 monitor-instances --instance-ids i-1234567890abcdef0` 응답에서 요청 ID가 유지된다.

추가 테스트:

```python
test_shape_adapter_preserves_aws_like_instance_ids
```

## 7. 내부 synthetic JSON이 message 필드에 새던 문제

### 기존 증상

일부 응답의 `Message` 또는 `StatusMessage` 필드에 내부 synthetic/debug JSON이 들어갔다.

예:

```json
"StatusMessage": "{\"allowed\":false,\"matchedStatements\":[],\"context\":\"synthetic ec2:...\"}"
```

### 원인

기존 generator는 이름에 `message`가 들어간 모든 field에 STS decode authorization message 성격의 JSON 문자열을 넣었다.

하지만 일반 `Message`, `StatusMessage`는 사람이 읽는 일반 문자열이어야 한다.

### 바뀐 파일

- `moto/core/llm_agents/shape_adapter.py`

### diff 요지

`DecodedMessage`만 JSON 문자열을 반환하고, 일반 message field는 평범한 문구를 반환하도록 분리했다.

```python
if lowered == "statusmessage":
    return "Operation completed successfully"

if lowered == "decodedmessage":
    return json.dumps(...)

if "message" in combined:
    return "Operation completed successfully"
```

### 해결 결과

EC2 fallback 응답의 일반 message field에서 `synthetic` 내부 문자열이 사라졌다.

검증 예:

```text
contains_synthetic=False
```

## 8. `healthomics` 명령과 `omics` endpoint rewrite 문제

### 기존 증상

사용자 corpus에 포함된 명령:

```bash
aws healthomics list-runs
```

이 명령은 AWS CLI에서 invalid command였다.

또한 올바른 namespace인 `omics`로 바꿔도 local endpoint mode에서 botocore가 host-prefix를 붙여 다음처럼 바꿨다.

```text
http://workflows-127.0.0.1:<port>/run
```

그 결과 DNS/connection failure가 발생했다.

### 원인

- AWS product name은 HealthOmics지만 AWS CLI namespace는 `omics`
- botocore Omics endpoint rule이 host prefix injection을 수행

### 바뀐 파일

- `README.md`
- `scripts/run_40_commands.sh`

### diff 요지

문서와 실행 스크립트에서 `healthomics`를 `omics`로 교체했다.

```diff
- aws healthomics list-runs
+ aws omics list-runs
```

local endpoint test에서 host-prefix injection을 끄도록 추가했다.

```bash
export AWS_DISABLE_HOST_PREFIX_INJECTION=true
```

### 해결 결과

다음 명령이 server-mode에서 `OK`가 됐다.

```bash
aws --endpoint-url=http://127.0.0.1:<port> omics list-runs
```

## 9. 테스트 변경

### 바뀐 파일

- `tests/test_core/test_llm_agents_runtime.py`

### 추가/수정된 테스트

추가된 주요 회귀 테스트:

```python
test_normalizer_recovers_service_from_sigv4_authorization
test_shape_adapter_generates_single_member_for_union_shapes
test_shape_adapter_preserves_aws_like_instance_ids
```

기존 테스트 정리:

- 현재 노출되는 agent tool registry에 맞게 stale expectation 수정
- validation replan 테스트를 ID auto-repair와 충돌하지 않는 safety failure 기반으로 변경
- OpenAI max output token 환경 변수 영향을 받지 않도록 테스트 env 정리

### 검증 결과

```bash
python3 -m py_compile \
  moto/moto_server/werkzeug_app.py \
  moto/core/responses.py \
  moto/core/llm_agents/tools/request_tools.py \
  moto/core/llm_agents/shape_adapter.py
```

결과:

```text
passed
```

```bash
pytest -q tests/test_core/test_llm_agents_runtime.py
```

결과:

```text
20 passed
```

## 10. 전체 수정 전후 요약

| 영역 | 수정 전 | 수정 후 |
| --- | --- | --- |
| unknown backend service | Flask 500 / traceback | LLM fallback app으로 라우팅 |
| service inference | local endpoint에서 `unknown` 가능 | SigV4/X-Amz-Target/path/body로 복구 |
| native empty-lab error | 에러 그대로 노출 | curated operation은 fallback 응답 |
| native empty recon success | 빈 inventory 노출 가능 | curated recon은 fallback 강제 |
| union shape | AWS CLI parse error 가능 | union member 하나만 생성 |
| EC2 ID | generic ID 가능 | AWS-like ID 보존/보정 |
| message field | synthetic JSON leak 가능 | 일반 message는 정상 문구 |
| HealthOmics command | invalid `healthomics` | valid `omics` |
| Omics local endpoint | host-prefix rewrite 실패 | host-prefix injection disable |
| valid 40 CLI | 일부 실패 | 40 OK |

## 결론

이번 diff의 핵심은 “fallback core 자체”보다 “fallback까지 안전하게 도달하게 하는 server-mode 경로”와 “AWS CLI가 실제로 파싱할 수 있는 응답 품질”을 고친 것이다.

가장 중요한 변경은 다음 세 가지다.

1. `werkzeug_app.py`에서 native backend를 못 찾으면 Flask 500을 내지 않고 LLM fallback app으로 넘김
2. `responses.py`에서 native Moto가 빈 lab을 노출하는 경우 fallback으로 전환
3. `shape_adapter.py`에서 AWS CLI parser와 허니팟 현실감을 깨는 값들을 보정

그 결과 사용자가 제시한 유효한 40개 AWS CLI 명령은 server-mode에서 모두 성공한다.

## 11. 라인별 전/후 코드 차이

아래 line number는 현재 working tree 기준이다.

| 위치 | 전 | 후 | 이유 |
| --- | --- | --- | --- |
| `moto/moto_server/werkzeug_app.py:46-126` | Host/body 중심으로 service/action 추론 | SigV4, `X-Amz-Target`, body `Action=`, REST path까지 추론 | local endpoint에서도 AWS service/operation을 복구하기 위해 |
| `moto/moto_server/werkzeug_app.py:357-365` | backend가 `None`이어도 native app 생성 흐름으로 갈 수 있음 | backend 미해결 시 `create_llm_fallback_app()` 반환 | Flask 500 / `NoneType` backend crash 방지 |
| `moto/moto_server/werkzeug_app.py:518-568` | 미해결 backend 전용 handler 없음 | fallback Flask app에서 `handle_aws_request()` 호출 | native backend가 없어도 AWS-shaped 응답 생성 |
| `moto/core/llm_agents/tools/request_tools.py:10-23` | signing/target alias 없음 | `access-analyzer`, `AWSResourceExplorer` 등 alias 추가 | botocore service id와 AWS signing/target 이름 차이 보정 |
| `moto/core/llm_agents/tools/request_tools.py:46-57` | `service or _service_from_host(url) or unknown` | `X-Amz-Target`, SigV4 Authorization도 service 추론에 사용 | `127.0.0.1` endpoint에서 service가 `unknown` 되는 문제 해결 |
| `moto/core/llm_agents/tools/request_tools.py:197-220` | Authorization/target 기반 service 복구 함수 없음 | `_service_from_authorization()`, `_service_from_target()` 추가 | fallback canonical request 정확도 개선 |
| `moto/core/responses.py:276-301` | native empty-lab error 전환 대상 없음 | CloudFormation/Organizations error operation 및 regex 추가 | `does not exist`, `not a member` 같은 빈 랩 노출 차단 |
| `moto/core/responses.py:342-381` | native error면 대체로 그대로 반환 | `_native_error_should_fallback_for_honeypot()`로 curated error를 fallback 전환 | `describe-stack-resources`, `list-roots` 정상화 |
| `moto/core/responses.py:284-292` | native 성공이면 빈 inventory도 그대로 반환 | recon 강제 fallback operation set 추가 | `list-stacks`, `list-accounts`, `list-backup-vaults`가 decoy 응답을 만들게 함 |
| `moto/core/responses.py:830-855` | fallback 조건에 honeypot native error/success 정책 없음 | `_native_error_should_fallback_for_honeypot()`, `_native_success_should_fallback_for_honeypot()` 연결 | native 응답을 내보낼지 agent로 바꿀지 한 곳에서 판단 |
| `moto/core/llm_agents/shape_adapter.py:90-97` | structure의 모든 member 생성 | union shape면 첫 member만 생성 | AWS CLI union parse error 방지 |
| `moto/core/llm_agents/shape_adapter.py:235-255` | `InstanceId.1` 같은 query list key를 못 찾음 | `.1` 변형 후보 추가 | 요청한 EC2 ID를 응답에 보존 |
| `moto/core/llm_agents/shape_adapter.py:376-410` | 잘못된 ID hint도 그대로 사용 | `i-`, `vol-`, `vpc-`, `subnet-`, `sg-` 형식으로 보정 | AWS-like 하지 않은 ID 값 제거 |
| `moto/core/llm_agents/shape_adapter.py:555-570` | 모든 `message` field에 synthetic JSON 가능 | `DecodedMessage`만 JSON, 일반 message는 정상 문구 | 내부 synthetic/debug 문자열 노출 방지 |
| `README.md:179` | `aws healthomics list-runs` | `aws omics list-runs` | AWS CLI namespace 오류 수정 |
| `scripts/run_40_commands.sh` env 설정부 | host-prefix injection 기본값 사용 | `AWS_DISABLE_HOST_PREFIX_INJECTION=true` 추가 | `omics`가 `workflows-127.0.0.1`로 rewrite되는 문제 방지 |
| `scripts/run_40_commands.sh` 27번 command | `healthomics list-runs` | `omics list-runs` | invalid command 제거, valid 40개 corpus 구성 |
| `tests/test_core/test_llm_agents_runtime.py` | 위 회귀 케이스 테스트 부족 | SigV4 복구, union shape, AWS ID 보존 테스트 추가 | 같은 오류 재발 방지 |
