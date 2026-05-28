from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Callable

from . import honeypot_profile as profile
from .tools.request_tools import CanonicalRequest

_ACCOUNT_RE = re.compile(r"(?<=:)\d{12}(?=:)")
_ARN_ACCOUNT_RE = re.compile(r"^(arn:[^:]+:[^:]*:[^:]*:)(\d{12})(:.*)$")
_PLACEHOLDER_VALUES = {
    "description",
    "Description",
    "status",
    "StatusReason",
    "eventTypeName",
    "lastUpdatedTime",
    "createdTime",
    "owner",
    "Owner",
    "createdBy",
    "lastUpdatedBy",
    "sourceConnectorLabel",
    "destinationConnectorLabel",
    "GatewayDisplayName",
    "complianceType",
}


def apply_honeypot_profile_overrides(
    canonical: CanonicalRequest,
    payload: dict[str, Any],
    world_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    hp = world_state.get("honeypot_profile")
    if not isinstance(hp, dict) or not hp:
        return payload, {"profile_overrides": False}

    patched = deepcopy(payload)
    _rewrite_common(patched, canonical)

    handler = _OPERATION_OVERRIDES.get((canonical.service, canonical.operation))
    if handler is not None:
        handler(patched, canonical)

    _dedupe_lists(patched)
    return patched, {"profile_overrides": True}


def _rewrite_common(value: Any, canonical: CanonicalRequest, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in list(value.items()):
            if isinstance(child, str):
                value[child_key] = _rewrite_string(child_key, child, canonical)
            elif isinstance(child, (dict, list)):
                _rewrite_common(child, canonical, child_key)
            elif child_key.lower() in {
                "account",
                "accountid",
                "ownerid",
                "registryid",
                "primaryaccountid",
            }:
                value[child_key] = profile.ACCOUNT_ID
    elif isinstance(value, list):
        for item in value:
            _rewrite_common(item, canonical, key)


def _rewrite_string(key: str, text: str, canonical: CanonicalRequest) -> str:
    if text in _PLACEHOLDER_VALUES:
        return _fallback_named_value(key, canonical)

    lowered_key = key.lower()
    if lowered_key in {"account", "accountid", "ownerid", "registryid"}:
        return profile.ACCOUNT_ID
    if lowered_key in {"region", "availabilityzone"} and text == "region":
        return profile.REGION
    if lowered_key in {"username", "serviceusername"} and text in {
        "test",
        "moto",
        "victim-admin",
        "default_user",
    }:
        return profile.IAM_USER
    if lowered_key in {"repositoryname", "name"} and text == "demo":
        return profile.ECR_REPOSITORIES[0]
    if lowered_key in {"bucket", "bucketname"} and text in {
        "my-honeypot-bucket",
        "bucket",
    }:
        return profile.BACKUP_BUCKET
    if lowered_key in {"cluster", "clustername"} and text in {
        "test-cluster",
        "prod-cluster",
    }:
        return profile.EKS_CLUSTER

    if text.startswith("arn:"):
        text = _rewrite_arn(text)
    text = text.replace("123456789012", profile.ACCOUNT_ID)
    text = text.replace("victim-admin", profile.IAM_USER)
    text = text.replace("test-cluster", profile.EKS_CLUSTER)
    text = text.replace("prod-cluster", profile.EKS_CLUSTER)
    text = text.replace("my-honeypot-bucket", profile.BACKUP_BUCKET)
    if canonical.service == "ecr":
        text = text.replace("/demo", f"/{profile.ECR_REPOSITORIES[0]}")
    return text


def _rewrite_arn(arn: str) -> str:
    match = _ARN_ACCOUNT_RE.match(arn)
    if match:
        arn = f"{match.group(1)}{profile.ACCOUNT_ID}{match.group(3)}"
    return _ACCOUNT_RE.sub(profile.ACCOUNT_ID, arn)


def _fallback_named_value(key: str, canonical: CanonicalRequest) -> str:
    lowered = key.lower()
    prefix = profile.COMPANY_PREFIX
    if "description" in lowered:
        return f"{prefix} production {canonical.service} resource"
    if "reason" in lowered:
        return "Active in production"
    if "createdby" in lowered or "updatedby" in lowered or "owner" == lowered:
        return profile.IAM_USER
    if "gateway" in lowered:
        return f"{prefix}-prod-backup-gateway"
    if "flow" in lowered:
        return f"{prefix}-salesforce-billing-sync"
    if "compliance" in lowered:
        return "SOC2"
    return f"{prefix}-{canonical.service}-{canonical.operation.lower()}"


def _dedupe_lists(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _dedupe_lists(child)
    elif isinstance(value, list):
        seen: set[str] = set()
        deduped: list[Any] = []
        for item in value:
            marker = json.dumps(item, sort_keys=True, default=str)
            if marker not in seen:
                seen.add(marker)
                deduped.append(item)
        value[:] = deduped
        for child in value:
            _dedupe_lists(child)


def _override_bedrock_list_foundation_models(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    summaries = payload.get("modelSummaries")
    if not isinstance(summaries, list) or not summaries:
        summaries = [{}]
    models = [
        ("amazon.titan-text-premier-v1:0", "Amazon Titan Text Premier", "Amazon"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "Claude 3.5 Sonnet", "Anthropic"),
    ]
    payload["modelSummaries"] = [
        {
            **(
                summaries[idx]
                if idx < len(summaries) and isinstance(summaries[idx], dict)
                else {}
            ),
            "modelArn": f"arn:aws:bedrock:{profile.REGION}::foundation-model/{model_id}",
            "modelId": model_id,
            "modelName": name,
            "providerName": provider,
            "inputModalities": ["TEXT"],
            "outputModalities": ["TEXT"],
            "responseStreamingSupported": True,
            "customizationsSupported": [],
            "inferenceTypesSupported": ["ON_DEMAND"],
        }
        for idx, (model_id, name, provider) in enumerate(models)
    ]


def _override_ec2_monitor(payload: dict[str, Any], canonical: CanonicalRequest) -> None:
    target = (
        canonical.request_params.get("InstanceId.1")
        or canonical.request_params.get("InstanceIds")
        or canonical.request_params.get("instanceIds")
        or profile.INSTANCE_ID
    )
    if isinstance(target, list):
        target = target[0] if target else profile.INSTANCE_ID
    items = _find_first_list(payload, "instancesSet") or _find_first_list(
        payload, "InstanceMonitoring"
    )
    if not items:
        payload["InstanceMonitoring"] = [
            {"InstanceId": str(target), "Monitoring": {"State": "enabled"}}
        ]
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        _set_first_existing(item, ["InstanceId", "instanceId"], str(target))
        monitoring = item.setdefault("Monitoring", item.get("monitoring", {}))
        if isinstance(monitoring, dict):
            _set_first_existing(monitoring, ["State", "state"], "enabled")


def _override_ec2_describe_reserved_instances(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    items = _find_first_list(payload, "ReservedInstances") or _find_first_list(
        payload, "reservedInstancesSet"
    )
    if not items:
        return
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item["ReservedInstancesId"] = f"nexora-ri-{idx + 1:02d}"
        item["InstanceType"] = "m6i.large"
        item["AvailabilityZone"] = f"{profile.REGION}a"
        item["CurrencyCode"] = "USD"
        item["State"] = "active"


def _override_ec2_describe_volume_status(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    items = _find_first_list(payload, "VolumeStatuses") or _find_first_list(
        payload, "volumeStatusSet"
    )
    target = (
        canonical.request_params.get("VolumeId.1")
        or canonical.request_params.get("VolumeIds")
        or canonical.request_params.get("volumeIds")
        or "vol-0nexoraprod001"
    )
    if isinstance(target, list):
        target = target[0] if target else "vol-0nexoraprod001"
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item["VolumeId"] = str(target)
        item["AvailabilityZone"] = f"{profile.REGION}a"
        item["VolumeStatus"] = {"Status": "ok"}
        item["Events"] = []
        item["Actions"] = []


def _override_named_collection(
    payload: dict[str, Any],
    collection_key: str,
    name_key: str,
    arn_key: str,
    names: list[str],
    arn_resource: str,
) -> None:
    items = payload.get(collection_key)
    if not isinstance(items, list) or not items:
        items = [{} for _ in names]
    patched = []
    for idx, name in enumerate(names):
        base = items[idx] if idx < len(items) and isinstance(items[idx], dict) else {}
        base[name_key] = name
        base[arn_key] = (
            f"arn:aws:{arn_resource}:{profile.REGION}:{profile.ACCOUNT_ID}:{name}"
        )
        patched.append(base)
    payload[collection_key] = patched
    for token_key in ("NextToken", "nextToken"):
        if token_key in payload:
            payload[token_key] = ""


def _override_billing_groups(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    _override_named_collection(
        payload,
        "BillingGroups",
        "Name",
        "Arn",
        ["nexora-prod-billing-group", "nexora-payment-cost-center"],
        "billingconductor",
    )
    for item in payload.get("BillingGroups", []):
        if isinstance(item, dict):
            item["PrimaryAccountId"] = profile.ACCOUNT_ID
            item["Description"] = "Nexora production billing allocation"
            item["StatusReason"] = "Active production billing group"
            item["ComputationPreference"] = {
                "PricingPlanArn": (
                    f"arn:aws:billingconductor:{profile.REGION}:{profile.ACCOUNT_ID}:"
                    "pricingplan/nexora-prod-pricing-plan"
                )
            }
            item["AccountGrouping"] = {
                "AutoAssociate": False,
                "ResponsibilityTransferArn": (
                    f"arn:aws:billingconductor:{profile.REGION}:{profile.ACCOUNT_ID}:"
                    "billinggroup/nexora-prod-billing-group"
                ),
            }


def _override_frauddetector(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    detectors = payload.get("detectors")
    if not isinstance(detectors, list) or not detectors:
        detectors = [{}]
    payload["detectors"] = [
        {
            **(detectors[0] if isinstance(detectors[0], dict) else {}),
            "detectorId": "nexora-payment-fraud-detector",
            "description": "Nexora production payment fraud detector",
            "eventTypeName": "nexora-payment-checkout",
            "lastUpdatedTime": "2024-05-25T08:31:00Z",
            "createdTime": "2024-05-18T08:31:00Z",
            "arn": f"arn:aws:frauddetector:{profile.REGION}:{profile.ACCOUNT_ID}:detector/nexora-payment-fraud-detector",
        }
    ]
    payload["nextToken"] = ""


def _override_detective(payload: dict[str, Any], canonical: CanonicalRequest) -> None:
    payload["GraphList"] = [
        {
            "Arn": f"arn:aws:detective:{profile.REGION}:{profile.ACCOUNT_ID}:graph/nexora-prod-security-graph",
            "CreatedTime": "2024-05-18T08:31:00Z",
        }
    ]
    payload["NextToken"] = ""


def _override_auditmanager(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["assessmentMetadata"] = [
        {
            "name": "nexora-prod-soc2-assessment",
            "id": "nexora-soc2-prod",
            "complianceType": "SOC2",
            "status": "ACTIVE",
            "roles": [
                {
                    "roleType": "PROCESS_OWNER",
                    "roleArn": profile.BASTION_ROLE_ARN,
                }
            ],
            "delegations": [],
            "creationTime": 1716019200,
            "lastUpdated": 1716624000,
        },
        {
            "name": "nexora-payment-pci-assessment",
            "id": "nexora-pci-prod",
            "complianceType": "PCI-DSS",
            "status": "ACTIVE",
            "roles": [
                {
                    "roleType": "PROCESS_OWNER",
                    "roleArn": f"arn:aws:iam::{profile.ACCOUNT_ID}:role/nexora-payment-compliance",
                }
            ],
            "delegations": [],
            "creationTime": 1716019200,
            "lastUpdated": 1716624000,
        },
    ]
    payload["nextToken"] = ""


def _override_outposts(payload: dict[str, Any], canonical: CanonicalRequest) -> None:
    payload["Outposts"] = [
        {
            "OutpostId": "op-0nexoraprod01",
            "OwnerId": profile.ACCOUNT_ID,
            "OutpostArn": f"arn:aws:outposts:{profile.REGION}:{profile.ACCOUNT_ID}:outpost/op-0nexoraprod01",
            "SiteId": "os-0nexoraeast1",
            "Name": "nexora-prod-edge-rack",
            "Description": "Nexora production edge rack",
            "LifeCycleStatus": "ACTIVE",
            "AvailabilityZone": f"{profile.REGION}a",
            "AvailabilityZoneId": "use1-az1",
            "Tags": {
                "Environment": profile.ENVIRONMENT,
                "Company": profile.COMPANY_PREFIX,
            },
            "SupportedHardwareType": "RACK",
        }
    ]
    payload["NextToken"] = ""


def _override_appflow(payload: dict[str, Any], canonical: CanonicalRequest) -> None:
    payload["flows"] = [
        {
            "flowArn": f"arn:aws:appflow:{profile.REGION}:{profile.ACCOUNT_ID}:flow/nexora-salesforce-billing-sync",
            "description": "Syncs Salesforce account data into Nexora billing reports",
            "flowName": "nexora-salesforce-billing-sync",
            "flowStatus": "Active",
            "sourceConnectorType": "Salesforce",
            "destinationConnectorType": "S3",
            "triggerType": "Scheduled",
            "createdAt": 1716019200,
            "lastUpdatedAt": 1716624000,
            "createdBy": profile.IAM_USER,
            "lastUpdatedBy": profile.IAM_USER,
            "tags": {
                "Environment": profile.ENVIRONMENT,
                "Company": profile.COMPANY_PREFIX,
            },
        }
    ]
    payload["nextToken"] = ""


def _override_omics(payload: dict[str, Any], canonical: CanonicalRequest) -> None:
    payload["items"] = []
    payload["nextToken"] = ""


def _override_mgn(payload: dict[str, Any], canonical: CanonicalRequest) -> None:
    payload["items"] = [
        {
            "sourceServerID": "s-nexora-prod-bastion",
            "arn": f"arn:aws:mgn:{profile.REGION}:{profile.ACCOUNT_ID}:source-server/s-nexora-prod-bastion",
            "isArchived": False,
            "tags": {"Name": "nexora-prod-bastion", "Environment": profile.ENVIRONMENT},
            "launchedInstance": {"ec2InstanceID": profile.INSTANCE_ID},
            "dataReplicationState": "CONTINUOUS",
        }
    ]
    payload["nextToken"] = ""


def _override_codeguru(payload: dict[str, Any], canonical: CanonicalRequest) -> None:
    payload["RepositoryAssociationSummaries"] = [
        {
            "AssociationArn": f"arn:aws:codeguru-reviewer:{profile.REGION}:{profile.ACCOUNT_ID}:association:nexora-backend-api",
            "AssociationId": "nexora-backend-api",
            "Name": "nexora-backend-api",
            "Owner": profile.COMPANY_PREFIX,
            "ProviderType": "CodeCommit",
            "State": "Associated",
            "LastUpdatedTimeStamp": 1716624000,
        }
    ]
    payload["NextToken"] = ""


def _override_backup_gateway(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["Gateways"] = [
        {
            "GatewayArn": f"arn:aws:backup-gateway:{profile.REGION}:{profile.ACCOUNT_ID}:gateway/nexora-prod-backup-gateway",
            "GatewayDisplayName": "nexora-prod-backup-gateway",
            "GatewayType": "BACKUP_VM",
            "HypervisorId": "nexora-prod-vmware",
            "LastSeenTime": 1716624000,
        }
    ]
    payload["NextToken"] = ""


def _override_ssm_describe_instance_information(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["InstanceInformationList"] = [
        {
            "InstanceId": profile.INSTANCE_ID,
            "PingStatus": "Online",
            "LastPingDateTime": 1716624000,
            "AgentVersion": "3.2.700.0",
            "IsLatestVersion": True,
            "PlatformType": "Linux",
            "PlatformName": profile.OS_RELEASE,
            "PlatformVersion": "22.04",
            "ResourceType": "EC2Instance",
            "IPAddress": profile.PRIVATE_IP,
            "ComputerName": profile.HOSTNAME,
            "AssociationStatus": "Success",
            "IamRole": profile.INSTANCE_PROFILE,
        }
    ]
    payload["NextToken"] = ""


def _override_ecr_batch_check_layer_availability(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    digest = (
        canonical.request_params.get("layerDigest")
        or canonical.request_params.get("layerDigests")
        or canonical.request_params.get("LayerDigest.1")
        or "sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"
    )
    if isinstance(digest, list):
        digest = (
            digest[0]
            if digest
            else "sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"
        )
    payload["layers"] = [
        {
            "layerDigest": str(digest),
            "layerAvailability": "AVAILABLE",
            "layerSize": 5242880,
            "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
        }
    ]
    payload["failures"] = []


def _override_ecr_complete_layer_upload(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["registryId"] = profile.ACCOUNT_ID
    payload["repositoryName"] = profile.ECR_REPOSITORIES[0]
    if "uploadId" not in payload:
        payload["uploadId"] = str(
            canonical.request_params.get("uploadId") or "upload-nexora-prod"
        )
    if "layerDigest" not in payload:
        payload["layerDigest"] = str(
            canonical.request_params.get("layerDigest")
            or canonical.request_params.get("LayerDigest.1")
            or "sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"
        )


def _override_iam_context_keys(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["ContextKeyNames"] = ["aws:RequestedRegion", "aws:PrincipalArn"]


def _override_iam_list_service_specific(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    user_name = str(
        _request_param_ci(canonical, "UserName")
        or _request_param_ci(canonical, "userName")
        or canonical.target_identifiers.get("UserName")
        or canonical.target_identifiers.get("userName")
        or canonical.target_identifiers.get("user")
        or profile.IAM_USER
    )
    if user_name in {"victim-admin", "test", "moto"}:
        user_name = profile.IAM_USER
    payload["ServiceSpecificCredentials"] = [
        {
            "UserName": user_name,
            "Status": "Active",
            "ServiceUserName": user_name,
            "ServiceCredentialAlias": "codecommit-nexora-prod",
            "CreateDate": "2024-05-18T08:31:00Z",
            "ExpirationDate": "2026-05-18T08:31:00Z",
            "ServiceSpecificCredentialId": "ACCAEXAMPLENEXORA001",
            "ServiceName": "codecommit.amazonaws.com",
        }
    ]
    payload["IsTruncated"] = False


def _override_iam_generate_last_accessed(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["JobId"] = "job-nexora-prod-accessed"


def _override_secrets_validate_policy(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    raw_policy = str(
        canonical.request_params.get("ResourcePolicy")
        or canonical.request_params.get("resourcePolicy")
        or ""
    )
    broad = '"Principal":"*"' in raw_policy.replace(
        " ", ""
    ) or '"AWS":"*"' in raw_policy.replace(" ", "")
    if broad:
        payload["PolicyValidationPassed"] = False
        payload["ValidationErrors"] = [
            {
                "CheckName": "SECURITY_WARNING",
                "ErrorMessage": "Resource policy allows broad principal access to secret values.",
            }
        ]
    else:
        payload["PolicyValidationPassed"] = True
        payload["ValidationErrors"] = []


def _override_secrets_list_secrets(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["SecretList"] = [
        {
            "ARN": (
                f"arn:aws:secretsmanager:{profile.REGION}:{profile.ACCOUNT_ID}:"
                "secret:nexora/prod/db/password-a1b2c3"
            ),
            "Name": "nexora/prod/db/password",
            "Description": "Nexora production database credential",
            "KmsKeyId": (
                f"arn:aws:kms:{profile.REGION}:{profile.ACCOUNT_ID}:"
                "key/nexora-prod-secrets"
            ),
            "RotationEnabled": True,
            "RotationRules": {"AutomaticallyAfterDays": 30},
            "LastChangedDate": "2024-05-18T08:31:00Z",
            "LastAccessedDate": "2024-05-25T08:31:00Z",
            "NextRotationDate": "2024-06-17T08:31:00Z",
            "Tags": [
                {"Key": "Environment", "Value": profile.ENVIRONMENT},
                {"Key": "Company", "Value": profile.COMPANY_PREFIX},
                {"Key": "Owner", "Value": "platform"},
            ],
            "CreatedDate": "2024-05-18T08:31:00Z",
            "PrimaryRegion": profile.REGION,
        },
        {
            "ARN": (
                f"arn:aws:secretsmanager:{profile.REGION}:{profile.ACCOUNT_ID}:"
                "secret:nexora/payment/stripe/api-key-d4e5f6"
            ),
            "Name": "nexora/payment/stripe/api-key",
            "Description": "Nexora payment processor API key",
            "KmsKeyId": (
                f"arn:aws:kms:{profile.REGION}:{profile.ACCOUNT_ID}:"
                "key/nexora-prod-secrets"
            ),
            "RotationEnabled": True,
            "RotationRules": {"AutomaticallyAfterDays": 45},
            "LastChangedDate": "2024-05-20T08:31:00Z",
            "LastAccessedDate": "2024-05-25T08:31:00Z",
            "NextRotationDate": "2024-07-04T08:31:00Z",
            "Tags": [
                {"Key": "Environment", "Value": profile.ENVIRONMENT},
                {"Key": "Company", "Value": profile.COMPANY_PREFIX},
                {"Key": "Owner", "Value": "payments"},
            ],
            "CreatedDate": "2024-05-20T08:31:00Z",
            "PrimaryRegion": profile.REGION,
        },
    ]
    payload["NextToken"] = ""


def _override_sts_decode_authorization(
    payload: dict[str, Any], canonical: CanonicalRequest
) -> None:
    payload["DecodedMessage"] = json.dumps(
        {
            "allowed": False,
            "explicitDeny": False,
            "principal": profile.IAM_USER_ARN,
            "matchedStatements": [],
            "context": {
                "principalAccount": profile.ACCOUNT_ID,
                "principalArn": profile.IAM_USER_ARN,
                "region": profile.REGION,
            },
        },
        separators=(",", ":"),
    )


def _find_first_list(payload: Any, target_key: str) -> list[Any] | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == target_key and isinstance(value, list):
                return value
            found = _find_first_list(value, target_key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_list(item, target_key)
            if found is not None:
                return found
    return None


def _set_first_existing(payload: dict[str, Any], keys: list[str], value: Any) -> None:
    for key in keys:
        if key in payload:
            payload[key] = value
            return
    payload[keys[0]] = value


def _request_param_ci(canonical: CanonicalRequest, wanted: str) -> Any:
    wanted_lower = wanted.lower()
    for key, value in canonical.request_params.items():
        if key.lower() == wanted_lower:
            return value
    return None


_OPERATION_OVERRIDES: dict[
    tuple[str, str], Callable[[dict[str, Any], CanonicalRequest], None]
] = {
    ("bedrock", "ListFoundationModels"): _override_bedrock_list_foundation_models,
    ("ec2", "MonitorInstances"): _override_ec2_monitor,
    ("ec2", "DescribeReservedInstances"): _override_ec2_describe_reserved_instances,
    ("ec2", "DescribeVolumeStatus"): _override_ec2_describe_volume_status,
    ("billingconductor", "ListBillingGroups"): _override_billing_groups,
    ("frauddetector", "GetDetectors"): _override_frauddetector,
    ("detective", "ListGraphs"): _override_detective,
    ("auditmanager", "ListAssessments"): _override_auditmanager,
    ("outposts", "ListOutposts"): _override_outposts,
    ("appflow", "ListFlows"): _override_appflow,
    ("omics", "ListRuns"): _override_omics,
    ("healthomics", "ListRuns"): _override_omics,
    ("mgn", "DescribeSourceServers"): _override_mgn,
    ("codeguru-reviewer", "ListRepositoryAssociations"): _override_codeguru,
    ("backup-gateway", "ListGateways"): _override_backup_gateway,
    ("ssm", "DescribeInstanceInformation"): _override_ssm_describe_instance_information,
    (
        "ecr",
        "BatchCheckLayerAvailability",
    ): _override_ecr_batch_check_layer_availability,
    ("ecr", "CompleteLayerUpload"): _override_ecr_complete_layer_upload,
    ("iam", "GetContextKeysForPrincipalPolicy"): _override_iam_context_keys,
    ("iam", "ListServiceSpecificCredentials"): _override_iam_list_service_specific,
    ("iam", "GenerateServiceLastAccessedDetails"): _override_iam_generate_last_accessed,
    ("secretsmanager", "ListSecrets"): _override_secrets_list_secrets,
    ("secretsmanager", "ValidateResourcePolicy"): _override_secrets_validate_policy,
    ("sts", "DecodeAuthorizationMessage"): _override_sts_decode_authorization,
}
