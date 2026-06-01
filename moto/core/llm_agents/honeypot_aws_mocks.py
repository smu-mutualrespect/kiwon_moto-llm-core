"""Static AWS CLI responses for the Nexora production honeypot profile."""

from __future__ import annotations

import hashlib
from typing import Any

from . import honeypot_profile as profile

HONEYPOT_ACCESS_KEYS = {
    profile.DEFAULT_ACCESS_KEY_ID,
    profile.PROD_ACCESS_KEY_ID,
}


def is_honeypot_access_key(access_key_id: str) -> bool:
    return access_key_id in HONEYPOT_ACCESS_KEYS


def access_denied_message(action: str, resource: str = "*") -> str:
    return (
        f"User: {profile.IAM_USER_ARN} is not authorized to perform: {action} "
        f"on resource: {resource}"
    )


def iam_get_user() -> dict[str, Any]:
    return {
        "User": {
            "Path": "/",
            "UserName": profile.IAM_USER,
            "UserId": profile.CALLER_USER_ID,
            "Arn": profile.IAM_USER_ARN,
            "CreateDate": "2024-04-21T07:18:31Z",
            "Tags": [
                {"Key": "Environment", "Value": profile.ENVIRONMENT},
                {"Key": "Company", "Value": profile.COMPANY_PREFIX},
                {"Key": "AccessLevel", "Value": "operations"},
            ],
        }
    }


def iam_list_users() -> dict[str, Any]:
    return {
        "Users": [
            {
                "Path": "/",
                "UserName": user,
                "UserId": (
                    profile.CALLER_USER_ID
                    if user == profile.IAM_USER
                    else profile.stable_id("AIDA", f"iam-user:{user}", 16)
                ),
                "Arn": f"arn:aws:iam::{profile.ACCOUNT_ID}:user/{user}",
                "CreateDate": f"2024-04-{10 + idx:02d}T07:18:31Z",
            }
            for idx, user in enumerate(IAM_USERS)
        ],
        "IsTruncated": False,
    }


def iam_list_service_specific_credentials(user_name: str) -> dict[str, Any]:
    if user_name not in IAM_USERS:
        return {"ServiceSpecificCredentials": [], "IsTruncated": False}
    if user_name in {profile.IAM_USER, "ci-deploy", "terraform-cloud"}:
        return {
            "ServiceSpecificCredentials": [
                {
                    "UserName": user_name,
                    "Status": "Active",
                    "ServiceUserName": user_name,
                    "ServiceCredentialAlias": f"codecommit-{profile.COMPANY_PREFIX}-prod",
                    "CreateDate": "2024-05-18T08:31:00Z",
                    "ExpirationDate": "2026-05-18T08:31:00Z",
                    "ServiceSpecificCredentialId": (
                        "ACCAEXAMPLENEXORA"
                        f"{int(hashlib.sha1(user_name.encode()).hexdigest()[:3], 16) % 1000:03d}"
                    ),
                    "ServiceName": "codecommit.amazonaws.com",
                }
            ],
            "IsTruncated": False,
        }
    return {"ServiceSpecificCredentials": [], "IsTruncated": False}


def sensitive_s3_buckets() -> set[str]:
    return {
        profile.BACKUP_BUCKET,
        f"{profile.COMPANY_PREFIX}-prod-logs",
        f"{profile.COMPANY_PREFIX}-cloudtrail-main",
        f"{profile.COMPANY_PREFIX}-prod-terraform-state",
    }


def eks_list_clusters() -> dict[str, Any]:
    return {"clusters": [profile.EKS_CLUSTER], "nextToken": ""}


def eks_describe_cluster() -> dict[str, Any]:
    return {
        "cluster": {
            "name": profile.EKS_CLUSTER,
            "arn": f"arn:aws:eks:{profile.REGION}:{profile.ACCOUNT_ID}:cluster/{profile.EKS_CLUSTER}",
            "createdAt": 1714011784.0,
            "version": "1.28",
            "endpoint": f"https://A1B2C3D4E5F6.gr7.{profile.REGION}.eks.amazonaws.com",
            "roleArn": profile.EKS_ROLE_ARN,
            "resourcesVpcConfig": {
                "subnetIds": [
                    "subnet-0a4d7e912f0c8b611",
                    "subnet-0bd4291e83f8a6d0c",
                    "subnet-083f1b6a912e47d22",
                    "subnet-0dcb83192e7a4510a",
                ],
                "securityGroupIds": [profile.SECURITY_GROUP_ID],
                "clusterSecurityGroupId": "sg-08d2b3f4a91e6c0de",
                "vpcId": profile.VPC_ID,
                "endpointPublicAccess": False,
                "endpointPrivateAccess": True,
                "publicAccessCidrs": [],
            },
            "kubernetesNetworkConfig": {
                "serviceIpv4Cidr": "172.20.0.0/16",
                "ipFamily": "ipv4",
            },
            "logging": {
                "clusterLogging": [
                    {
                        "types": ["api", "audit", "authenticator"],
                        "enabled": True,
                    },
                    {"types": ["controllerManager", "scheduler"], "enabled": False},
                ]
            },
            "status": "ACTIVE",
            "certificateAuthority": {
                "data": profile.stable_b64(f"eks-ca:{profile.EKS_CLUSTER}", 88)
            },
            "platformVersion": "eks.18",
            "tags": {
                "Name": profile.EKS_CLUSTER,
                "Environment": profile.ENVIRONMENT,
                "Company": profile.COMPANY_PREFIX,
                "Owner": profile.IAM_USER,
                "VpcName": profile.VPC_NAME,
                "PrivateSubnetA": profile.PRIVATE_SUBNET_A_NAME,
                "PrivateSubnetB": profile.PRIVATE_SUBNET_B_NAME,
                "PublicSubnetA": profile.PUBLIC_SUBNET_A_NAME,
            },
        }
    }


def ecr_describe_repositories() -> dict[str, Any]:
    repositories = []
    for name in profile.ECR_REPOSITORIES:
        repositories.append(
            {
                "repositoryArn": f"arn:aws:ecr:{profile.REGION}:{profile.ACCOUNT_ID}:repository/{name}",
                "registryId": profile.ACCOUNT_ID,
                "repositoryName": name,
                "repositoryUri": f"{profile.ECR_REGISTRY}/{name}",
                "createdAt": 1713655500.0,
                "imageTagMutability": "MUTABLE",
                "imageScanningConfiguration": {"scanOnPush": True},
                "encryptionConfiguration": {"encryptionType": "AES256"},
            }
        )
    return {"repositories": repositories, "failures": []}


def lambda_list_functions() -> dict[str, Any]:
    names = [
        "nexora-alert-discord-lambda",
        "nexora-payment-webhook-lambda",
        "nexora-report-export-lambda",
        "nexora-prod-log-forwarder",
        "nexora-billing-retry-lambda",
    ]
    return {
        "Functions": [
            {
                "FunctionName": name,
                "FunctionArn": f"arn:aws:lambda:{profile.REGION}:{profile.ACCOUNT_ID}:function:{name}",
                "Runtime": "python3.10",
                "Role": f"arn:aws:iam::{profile.ACCOUNT_ID}:role/{name}-role",
                "Handler": "app.handler",
                "CodeSize": 1843912 + idx * 32817,
                "Description": f"{profile.COMPANY_PREFIX} production operations function",
                "Timeout": 30,
                "MemorySize": 256,
                "LastModified": f"2024-05-{18 + idx:02d}T08:13:4{idx}.000+0000",
                "CodeSha256": profile.stable_b64(f"lambda:{name}", 44),
                "Version": "$LATEST",
                "TracingConfig": {"Mode": "PassThrough"},
                "PackageType": "Zip",
                "Architectures": ["x86_64"],
                "EphemeralStorage": {"Size": 512},
            }
            for idx, name in enumerate(names)
        ]
    }


def logs_describe_log_groups() -> dict[str, Any]:
    names = [
        "/aws/eks/nexora-prod-eks/cluster",
        "/aws/lambda/nexora-alert-discord-lambda",
        "/aws/lambda/nexora-payment-webhook-lambda",
        "/aws/lambda/nexora-prod-log-forwarder",
        "/nexora/production/backend-api",
        "/nexora/production/nginx",
    ]
    return {
        "logGroups": [
            {
                "logGroupName": name,
                "creationTime": 1714011784000 + idx * 86400000,
                "retentionInDays": 30 if idx < 4 else 14,
                "metricFilterCount": 1 if idx in {0, 4, 5} else 0,
                "arn": f"arn:aws:logs:{profile.REGION}:{profile.ACCOUNT_ID}:log-group:{name}:*",
                "storedBytes": 7423912 + idx * 391024,
            }
            for idx, name in enumerate(names)
        ],
        "nextToken": None,
    }


def ssm_describe_instance_information() -> dict[str, Any]:
    return {
        "InstanceInformationList": [
            {
                "InstanceId": profile.INSTANCE_ID,
                "PingStatus": "Online",
                "LastPingDateTime": 1716710400.0,
                "AgentVersion": "3.3.1310.0",
                "IsLatestVersion": True,
                "PlatformType": "Linux",
                "PlatformName": "Ubuntu",
                "PlatformVersion": "22.04",
                "ResourceType": "EC2Instance",
                "IPAddress": profile.PRIVATE_IP,
                "ComputerName": profile.HOSTNAME,
                "AssociationStatus": "Success",
                "IamRole": profile.BASTION_ROLE_ARN.rsplit("/", 1)[-1],
                "RegistrationDate": 1713655500.0,
                "Name": f"{profile.COMPANY_PREFIX}-prod-bastion",
            }
        ]
    }


def secretsmanager_list_secrets() -> dict[str, Any]:
    secrets = [
        (
            "prod/db/password",
            "Primary RDS credential for backend-api",
            f"arn:aws:kms:{profile.REGION}:{profile.ACCOUNT_ID}:key/"
            f"{profile.stable_hex('kms:prod-secrets', 32)}",
            True,
        ),
        (
            "prod/payment/stripe",
            "Payment provider API credential",
            f"arn:aws:kms:{profile.REGION}:{profile.ACCOUNT_ID}:key/"
            f"{profile.stable_hex('kms:payment-secrets', 32)}",
            True,
        ),
        (
            "prod/ci/ecr-token",
            "CI token used by deployment automation",
            f"arn:aws:kms:{profile.REGION}:{profile.ACCOUNT_ID}:key/"
            f"{profile.stable_hex('kms:ci-secrets', 32)}",
            False,
        ),
    ]
    return {
        "SecretList": [
            {
                "ARN": (
                    f"arn:aws:secretsmanager:{profile.REGION}:{profile.ACCOUNT_ID}"
                    f":secret:{name}-{profile.stable_token(name, 6)}"
                ),
                "Name": name,
                "Description": description,
                "KmsKeyId": kms_key,
                "RotationEnabled": rotation_enabled,
                "RotationLambdaARN": (
                    f"arn:aws:lambda:{profile.REGION}:{profile.ACCOUNT_ID}"
                    f":function:{profile.COMPANY_PREFIX}-secret-rotation"
                ),
                "RotationRules": {
                    "AutomaticallyAfterDays": 30,
                    "Duration": "2h",
                    "ScheduleExpression": "rate(30 days)",
                },
                "LastChangedDate": 1715947200.0 + idx * 86400,
                "LastAccessedDate": 1716681600.0 + idx * 86400,
                "CreatedDate": 1713655500.0 + idx * 86400,
                "Tags": [
                    {"Key": "Environment", "Value": profile.ENVIRONMENT},
                    {"Key": "Company", "Value": profile.COMPANY_PREFIX},
                    {"Key": "Owner", "Value": "platform"},
                ],
                "SecretVersionsToStages": {
                    profile.stable_hex(f"secret-version:{name}", 32): ["AWSCURRENT"]
                },
                "PrimaryRegion": profile.REGION,
            }
            for idx, (name, description, kms_key, rotation_enabled) in enumerate(
                secrets
            )
        ]
    }


def validate_secret_resource_policy() -> dict[str, Any]:
    return {
        "PolicyValidationPassed": False,
        "ValidationErrors": [
            {
                "CheckName": "PublicPolicyCheck",
                "ErrorMessage": (
                    "Policy allows broad access to secretsmanager:GetSecretValue. "
                    "Restrict Principal and Resource before applying."
                ),
            }
        ],
    }


def s3_list_buckets() -> dict[str, Any]:
    buckets = [
        "nexora-prod-logs",
        "nexora-backup-prod",
        "nexora-prod-terraform-state",
        "nexora-prod-app-artifacts",
        "nexora-prod-alb-logs",
        "nexora-cloudtrail-main",
        "nexora-prod-static-assets",
        "nexora-payment-reports-prod",
    ]
    return {
        "Owner": {
            "ID": profile.stable_hex("s3-canonical-user-id", 64),
            "DisplayName": profile.COMPANY_PREFIX,
        },
        "Buckets": [
            {
                "Name": bucket,
                "CreationDate": f"2024-0{(idx % 5) + 1}-1{idx}T03:12:00.000Z",
            }
            for idx, bucket in enumerate(buckets)
        ],
    }


def ec2_describe_instances() -> dict[str, Any]:
    instances = [
        (
            profile.INSTANCE_ID,
            profile.HOSTNAME,
            "nexora-prod-bastion",
            "10.20.4.37",
            profile.SUBNET_ID,
            "t3.small",
        ),
        (
            "i-0741d92ef8c3b5a01",
            "ip-10-20-11-83",
            "nexora-prod-eks-node-a",
            "10.20.11.83",
            "subnet-0a4d7e912f0c8b611",
            "m6i.large",
        ),
        (
            "i-0b923fc18a6d472e5",
            "ip-10-20-12-41",
            "nexora-prod-eks-node-b",
            "10.20.12.41",
            "subnet-0bd4291e83f8a6d0c",
            "m6i.large",
        ),
        (
            "i-01d94b7a2c83e56f0",
            "ip-10-20-13-22",
            "nexora-prod-eks-node-c",
            "10.20.13.22",
            "subnet-083f1b6a912e47d22",
            "m6i.large",
        ),
    ]
    return {
        "Reservations": [
            {
                "ReservationId": f"r-{instance_id[2:]}",
                "OwnerId": profile.ACCOUNT_ID,
                "Groups": [],
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "ImageId": profile.AMI_ID,
                        "InstanceType": instance_type,
                        "KeyName": f"{profile.COMPANY_PREFIX}-prod-ops",
                        "LaunchTime": "2024-05-18T08:30:00.000Z",
                        "Placement": {
                            "AvailabilityZone": f"{profile.REGION}{chr(97 + idx)}",
                            "Tenancy": "default",
                        },
                        "PrivateDnsName": f"{hostname}.ec2.internal",
                        "PrivateIpAddress": private_ip,
                        "State": {"Code": 16, "Name": "running"},
                        "SubnetId": subnet_id,
                        "VpcId": profile.VPC_ID,
                        "SecurityGroups": [
                            {
                                "GroupId": profile.SECURITY_GROUP_ID,
                                "GroupName": f"{profile.COMPANY_PREFIX}-prod-bastion-sg"
                                if idx == 0
                                else f"{profile.COMPANY_PREFIX}-prod-node-sg",
                            }
                        ],
                        "IamInstanceProfile": {
                            "Arn": profile.BASTION_ROLE_ARN
                            if idx == 0
                            else profile.EKS_ROLE_ARN,
                            "Id": profile.stable_id(
                                "AIPA", f"instance-profile:{instance_id}", 17
                            ),
                        },
                        "Tags": [
                            {"Key": "Name", "Value": name},
                            {"Key": "Environment", "Value": profile.ENVIRONMENT},
                            {"Key": "Company", "Value": profile.COMPANY_PREFIX},
                            {"Key": "VpcName", "Value": profile.VPC_NAME},
                        ],
                    }
                ],
            }
            for idx, (
                instance_id,
                hostname,
                name,
                private_ip,
                subnet_id,
                instance_type,
            ) in enumerate(instances)
        ],
        "NextToken": None,
    }


IAM_USERS = [
    "devops-operator",
    "sre-oncall",
    "ci-deploy",
    "terraform-cloud",
    "backend-ci",
    "payment-ci",
    "readonly-auditor",
    "security-review",
    "data-pipeline",
    "support-readonly",
    "nexora-admin-breakglass",
    "billing-ops",
]
