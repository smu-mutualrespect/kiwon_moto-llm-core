# LLM Fallback CLI Honeypot Fix Plan

## Context

This document summarizes the May 15, 2026 test of the `origin/sub_main` LLM fallback implementation using the user-provided 40 AWS CLI commands.

Two test modes were used:

1. Direct runtime test: call `moto.core.llm_agents.agent.handle_aws_request()` directly for each command.
2. Real CLI integration test: run `moto_server`, then execute `aws --endpoint-url=http://127.0.0.1:<port> ...`.

The direct runtime path passed all 40 commands after using the correct session-derived account ID. The real CLI integration path did not.

## Test Results

### Direct Runtime

Result: `40/40` effective pass.

Observed properties:

- OpenAI provider calls succeeded for all commands.
- JSON/XML protocol matched expected AWS protocol for all commands.
- Required core response fields were present for all commands.
- Botocore top-level and recursive output shape checks passed for all commands.
- Average latency was about `1766ms`.
- `37/40` completed under 3 seconds.
- `37/40` completed under 4 seconds.

Initial validation appeared to be `21/40`, but that was a test harness issue: validation used account ID `123456789012` while the fallback session derived account ID `730115760625` from the credential. Re-evaluating with the actual session account ID produced `40/40`.

### Real AWS CLI Integration

Command shape:

```bash
aws --endpoint-url=http://127.0.0.1:5017 <service> <operation> ...
```

Result:

- Successful CLI rows: `23`
- Failed CLI rows: `18`
- The run included the original invalid `healthomics` command and an additional `omics` substituted check, so there were 41 rows total.
- LLM fallback audit records: `20`

Successful categories:

- `bedrock list-foundation-models`
- All tested EC2 commands
- `ssm describe-instance-information`
- All tested ECR commands
- All tested IAM commands
- `secretsmanager validate-resource-policy`
- `sts decode-authorization-message`
- Some Moto-native commands such as `cloudformation list-stacks`, `organizations list-accounts`, and `backup list-backup-vaults`

Failed categories:

- `resource-explorer-2`
- `accessanalyzer`
- `billingconductor`
- `frauddetector`
- `detective`
- `auditmanager`
- `outposts`
- `appflow`
- `mgn`
- `codeguru-reviewer`
- `backup-gateway`
- `cloudformation describe-stack-resources`
- `organizations list-roots`
- `healthomics list-runs`
- substituted `omics list-runs`

## Problem 1: Server Routing Fails Before Fallback

### Symptom

Many services return a Flask HTML `500 Internal Server Error` instead of an AWS-like fallback response.

Representative CLI error:

```text
An error occurred (500) when calling the ListIndexes operation:
<!doctype html>
<html lang=en>
<title>500 Internal Server Error</title>
```

Representative server traceback:

```text
AttributeError: 'NoneType' object has no attribute 'replace'
```

### Root Cause

In `moto/moto_server/werkzeug_app.py`, `get_application()` tries to infer a Moto backend from host/path/body. For several JSON protocol services, the backend is not inferred.

The unresolved backend is then passed into `create_backend_app()`, which calls:

```python
backend_dict = backends.get_backend(service)
```

In `moto/backends.py`, `get_backend()` assumes `name` is not `None`:

```python
safe_name = name.replace("-", "")
```

When `name is None`, the server crashes before `responses.py` or the LLM fallback path can run.

### Affected Commands

- `aws resource-explorer-2 list-indexes`
- `aws resource-explorer-2 list-views`
- `aws resource-explorer-2 search ...`
- `aws accessanalyzer list-analyzers`
- `aws accessanalyzer list-findings ...`
- `aws billingconductor list-billing-groups`
- `aws frauddetector get-detectors`
- `aws detective list-graphs`
- `aws auditmanager list-assessments`
- `aws outposts list-outposts`
- `aws appflow list-flows`
- `aws mgn describe-source-servers`
- `aws codeguru-reviewer list-repository-associations`
- `aws backup-gateway list-gateways`

### Fix Plan

1. Add a fallback service inference step in `werkzeug_app.py`.
   - Inspect SigV4 `Authorization` credential scope.
   - Inspect `X-Amz-Target`.
   - Inspect botocore-style request path.
   - Inspect AWS CLI user agent when useful.

2. If backend inference still fails, do not call `backends.get_backend(None)`.
   - Route the request to a small fallback Flask app/callback.
   - The callback should call `handle_aws_request(service=None, action=None, url=..., headers=..., body=..., source="moto_server.unresolved_backend")`.

3. Ensure fallback normalization can recover service/operation from:
   - `X-Amz-Target`
   - request path
   - body
   - host-style information

4. Add a regression test that a previously unknown server-mode JSON service returns a `200` AWS-like fallback body, not Flask HTML `500`.

## Problem 2: Native Moto Errors Leak Instead Of Falling Back

### Symptom

Some commands reach Moto native handlers, but Moto returns empty-state or not-found errors:

```text
Stack with id prod-app does not exist
```

```text
Your account is not a member of an organization.
```

These are plausible AWS errors, but they are bad for a deception environment because they collapse the fake environment too quickly.

### Affected Commands

- `aws cloudformation describe-stack-resources --stack-name prod-app`
- `aws organizations list-roots`

### Root Cause

The existing fallback trigger in `moto/core/responses.py` mainly catches:

- `NotImplementedError`
- missing action
- missing handler
- Moto native schema mismatch
- repeated agent operation cache
- session resource-not-found where the missing resource ID was previously exposed

It does not broadly treat "attacker reconnaissance command returned empty/not-in-use/not-found native state" as a fallback-worthy case.

### Fix Plan

1. Add a broader honeypot fallback decision helper in `responses.py`, for example:

```python
def _native_error_should_fallback_for_honeypot(
    session_id: str,
    service: str,
    operation: str | None,
    status: int,
    body: Any,
) -> bool:
    ...
```

2. Trigger fallback for selected reconnaissance operations when native Moto returns:
   - `ValidationError`
   - `NotFound`
   - `NoSuchEntity`
   - `AWSOrganizationsNotInUseException`
   - "does not exist"
   - "not a member of an organization"

3. Keep normal Moto tests safe by scoping this behavior.
   - Gate it behind an env var such as `MOTO_LLM_HONEYPOT_NATIVE_ERROR_FALLBACK=1`, or
   - activate only when LLM fallback is configured, or
   - activate only for a curated high-value operation set.

4. Add tests for:
   - `cloudformation:DescribeStackResources`
   - `organizations:ListRoots`

Expected result: the attacker receives AWS-like stack resources and organization roots instead of an empty lab error.

## Problem 3: `healthomics` Is Not A Valid AWS CLI Service Name

### Symptom

The command:

```bash
aws healthomics list-runs
```

fails before any network request:

```text
invalid choice 'healthomics'
```

### Root Cause

The installed AWS CLI exposes this service as `omics`, not `healthomics`.

### Follow-up Issue

Substituting:

```bash
aws omics list-runs
```

still failed in endpoint mode:

```text
Could not connect to the endpoint URL: "http://workflows-127.0.0.1:5017/run"
```

This appears to be a botocore endpoint-rule behavior for Omics workflows. The local endpoint URL is rewritten into a host prefix that does not resolve.

### Fix Plan

1. Replace corpus command `aws healthomics list-runs` with `aws omics list-runs`.
2. Add an endpoint-host workaround for Omics in the CLI test harness.
   - Use botocore config to disable host prefix injection if available.
   - Or add local host mapping for the generated host.
   - Or test Omics through direct runtime until server endpoint handling is patched.
3. Add a note to the corpus that the AWS product name is HealthOmics but the CLI namespace is `omics`.

## Problem 4: Some Successful Responses Are AWS-shaped But Semantically Weak

### Symptom

Some successful CLI responses are syntactically valid but contain weak fake values.

Examples:

```json
{
  "InstanceId": "instanceid-12345abcde"
}
```

AWS EC2 instance IDs should look like:

```text
i-1234567890abcdef0
```

Another example:

```json
{
  "StatusMessage": "{\"allowed\":false,\"matchedStatements\":[],\"context\":\"synthetic ec2:DescribeReservedInstancesListings\"}"
}
```

This is shape-valid, but it reads like synthetic internal metadata rather than normal AWS output.

### Root Cause

`moto/core/llm_agents/shape_adapter.py` generates values from botocore shapes plus generic field-name heuristics. Some field names are not covered by AWS-specific semantic rules, so they fall back to generic deterministic strings.

### Fix Plan

1. Expand semantic generation rules in `shape_adapter.py`.
2. Add service-specific field generation for:
   - EC2 IDs: `InstanceId`, `VolumeId`, `VpcId`, `SubnetId`, `SecurityGroupId`, `SnapshotId`, `ImageId`
   - EC2 status and state fields
   - ARNs
   - S3 bucket names
   - IAM user/role/resource names
   - CloudFormation stack names and stack IDs
   - Organizations account/root IDs
   - Access Analyzer analyzer ARNs
3. Prevent policy-analysis JSON from appearing in arbitrary string fields like `StatusMessage`.
4. Add value-format tests, not just shape tests.

Example assertions:

```python
assert re.match(r"^i-[0-9a-f]{8,17}$", instance_id)
assert "synthetic" not in status_message.lower()
assert not status_message.lstrip().startswith("{")
```

## Problem 5: Audit Count Shows Not All Commands Use Fallback

### Observation

The real CLI test produced only `20` LLM audit records even though 40 commands were issued.

This means:

- Some commands were handled by Moto native.
- Some commands crashed in server routing before fallback.
- Some native errors were returned directly.

### Fix Plan

1. Track three separate pass metrics:
   - CLI command succeeded.
   - LLM fallback was invoked.
   - Output passed deception-quality checks.

2. Add a test summary script that reports:
   - `cli_ok`
   - `fallback_invoked`
   - `native_ok`
   - `server_500`
   - `native_error_leak`
   - `shape_pass`
   - `semantic_pass`

3. Decide which operations should always be LLM-backed.
   - High-value reconnaissance and privilege-probing commands should prefer fallback even if Moto native can return an empty response.

## Recommended Implementation Order

### Phase 1: Stop Server 500 Leaks

Goal: no Flask HTML errors for AWS CLI commands.

Tasks:

1. Patch `moto_server/werkzeug_app.py` to avoid `backends.get_backend(None)`.
2. Add unresolved-backend fallback to `handle_aws_request()`.
3. Add tests for `resource-explorer-2`, `accessanalyzer`, and `billingconductor` server-mode requests.

Success criteria:

- Current `500 Internal Server Error` cases become AWS-like JSON responses.
- No traceback appears in CLI output.

### Phase 2: Route Empty Native Recon Errors To Fallback

Goal: avoid exposing an empty lab environment.

Tasks:

1. Add native error fallback helper in `responses.py`.
2. Add curated operation list for recon and cloud inventory commands.
3. Test CloudFormation and Organizations error cases.

Success criteria:

- `describe-stack-resources --stack-name prod-app` returns plausible stack resources.
- `organizations list-roots` returns plausible roots.

### Phase 3: Improve Semantic Fidelity

Goal: outputs should look like real AWS, not just botocore-shaped data.

Tasks:

1. Expand `shape_adapter.py` semantic value generation.
2. Add regex validation for core AWS identifiers.
3. Remove synthetic policy JSON from generic string fields.

Success criteria:

- EC2 IDs use real AWS ID formats.
- Status/message fields contain normal AWS-like text.
- No placeholders or internal words leak.

### Phase 4: Fix Corpus And CLI Harness

Goal: make the 40-command test reproducible.

Tasks:

1. Replace `healthomics` with `omics`.
2. Add Omics endpoint workaround or mark it direct-runtime-only until server mode supports it.
3. Save a canonical test script under `scripts/`.
4. Save result summary under `artifacts/agentic_runtime/`.

Success criteria:

- One command runs the full suite.
- Summary distinguishes direct runtime pass from actual CLI server-mode pass.

### Phase 5: Regression Gate

Goal: avoid future regressions.

Tasks:

1. Add a small server-mode CI subset:
   - `bedrock list-foundation-models`
   - `resource-explorer-2 list-indexes`
   - `accessanalyzer list-analyzers`
   - `cloudformation describe-stack-resources`
   - `organizations list-roots`
   - `ecr initiate-layer-upload`
   - `sts decode-authorization-message`

2. Check:
   - exit code is zero
   - body is parseable
   - no Flask HTML appears
   - no `NotImplemented`, traceback, or `llm_fallback!!`
   - required core fields exist

## Final Assessment

The LLM fallback core is suitable for honeypot deception. It can generate AWS-shaped responses across the 40-command set.

The current server-mode integration is not yet suitable for production honeypot use because several requests fail before fallback and some native Moto errors leak the empty mock environment.

The highest-priority fix is server-mode fallback routing. Once server routing no longer emits 500 HTML, the next priority is converting native empty-state errors into plausible honeypot responses.

## Implementation Status After Patch

The Phase 1 to Phase 3 fixes were implemented in this branch.

Implemented changes:

- `moto/moto_server/werkzeug_app.py`
  - Added SigV4 `Authorization` service/region inference.
  - Added `X-Amz-Target` prefix aliases for services whose host cannot be mapped directly.
  - Added unresolved-backend fallback routing instead of calling `backends.get_backend(None)`.
  - Added a fallback Flask app that invokes `handle_aws_request()` and returns AWS-like fallback bodies.

- `moto/core/llm_agents/tools/request_tools.py`
  - Added service recovery from `X-Amz-Target`.
  - Added service recovery from SigV4 credential scope.
  - Added alias handling for services such as `access-analyzer` -> `accessanalyzer`.

- `moto/core/responses.py`
  - Added curated native-error fallback for honeypot-sensitive empty-lab errors.
  - Currently covers `cloudformation:DescribeStackResources` and `organizations:ListRoots`.
  - Guarded by LLM fallback configuration and `MOTO_LLM_HONEYPOT_NATIVE_ERROR_FALLBACK`.

- `moto/core/llm_agents/shape_adapter.py`
  - Emits only one member for botocore union shapes.
  - Preserves query parameter hints such as `InstanceId.1`.
  - Normalizes core AWS IDs such as EC2 instance IDs and volume IDs.
  - Uses normal message text for generic message/status fields.
  - Keeps `DecodedMessage` specialized for STS authorization decode output.

- `tests/test_core/test_llm_agents_runtime.py`
  - Added regression coverage for SigV4 service inference.
  - Added regression coverage for union-shape rendering.
  - Added regression coverage for AWS-like instance ID preservation.
  - Updated stale agent-tool registry expectations to match the currently exposed tools.
  - Kept a validation-replan test using a safety-filter failure that is not auto-repaired by the semantic adapter.

Verification after patch:

- `python3 -m py_compile moto/moto_server/werkzeug_app.py moto/core/responses.py moto/core/llm_agents/tools/request_tools.py moto/core/llm_agents/shape_adapter.py`: passed.
- `pytest -q tests/test_core/test_llm_agents_runtime.py`: `20 passed`.
- Full CLI server-mode suite:
  - `40` successful valid command rows.
  - `1` failed invalid-command row in a 41-row comparison harness because the harness kept the original `healthomics` command beside the corrected `omics` command.
  - No server tracebacks in `/tmp/user40_moto_server.log`.
  - LLM fallback audit records: `40`.

Remaining compatibility notes:

- `aws healthomics list-runs` is not a valid AWS CLI command in the installed CLI; the CLI namespace is `omics`.
- `aws omics list-runs` requires `AWS_DISABLE_HOST_PREFIX_INJECTION=true` in local endpoint-mode testing. Without it, botocore rewrites the endpoint to a host-prefixed URL such as `http://workflows-127.0.0.1:<port>/run`.
- `scripts/run_40_commands.sh` now uses the corrected `omics` command and exports `AWS_DISABLE_HOST_PREFIX_INJECTION=true`.
