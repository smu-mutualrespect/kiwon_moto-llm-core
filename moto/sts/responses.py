from moto.core.llm_agents import honeypot_profile
from moto.core.responses import ActionResult, BaseResponse

from .exceptions import STSValidationError
from .models import STSBackend, sts_backends

MAX_FEDERATION_TOKEN_POLICY_LENGTH = 2048
MAX_ROLE_NAME_LENGTH = 64


class TokenResponse(BaseResponse):
    def __init__(self) -> None:
        super().__init__(service_name="sts")
        self.automated_parameter_parsing = True

    @property
    def backend(self) -> STSBackend:
        return sts_backends[self.current_account][self.partition]

    def _determine_resource(self) -> str:
        if "AssumeRole" in self.querystring.get("Action", []):
            return self.querystring.get("RoleArn")[0]  # type: ignore[index]
        return "*"

    def get_session_token(self) -> ActionResult:
        duration = self._get_int_param("DurationSeconds", 43200)
        token = self.backend.get_session_token(duration=duration)
        result = {
            "Credentials": {
                "AccessKeyId": honeypot_profile.stable_id(
                    "ASIA", "session-token-access-key", 16
                ),
                "SecretAccessKey": honeypot_profile.stable_b64(
                    "session-token-secret", 40
                ),
                "SessionToken": honeypot_profile.stable_b64("session-token-value", 256),
                "Expiration": token.expiration,
            }
        }
        return ActionResult(result)

    def get_federation_token(self) -> ActionResult:
        duration = self._get_int_param("DurationSeconds", 43200)
        policy = self._get_param("Policy")
        if policy is not None and len(policy) > MAX_FEDERATION_TOKEN_POLICY_LENGTH:
            raise STSValidationError(
                "1 validation error detected: Value "
                '\'{"Version": "2012-10-17", "Statement": [...]}\' '
                "at 'policy' failed to satisfy constraint: Member must have length less than or "
                f" equal to {MAX_FEDERATION_TOKEN_POLICY_LENGTH}"
            )
        name = self._get_param("Name")
        token = self.backend.get_federation_token(duration=duration, name=name)
        result = {
            "Credentials": {
                "AccessKeyId": honeypot_profile.stable_id(
                    "ASIA", f"federation-token-access-key:{name}", 16
                ),
                "SecretAccessKey": honeypot_profile.stable_b64(
                    f"federation-token-secret:{name}", 40
                ),
                "SessionToken": honeypot_profile.stable_b64(
                    f"federation-token-value:{name}", 256
                ),
                "Expiration": token.expiration,
            },
            "FederatedUser": {
                "FederatedUserId": f"{self.current_account}:{token.name}",
                "Arn": f"arn:{self.partition}:sts::{self.current_account}:federated-user/{token.name}",
            },
            "PackedPolicySize": 6,
        }
        return ActionResult(result)

    def assume_role(self) -> ActionResult:
        role_session_name = self._get_param("RoleSessionName")
        role_arn = self._get_param("RoleArn")
        policy = self._get_param("Policy")
        duration = self._get_int_param("DurationSeconds", 3600)
        external_id = self._get_param("ExternalId")
        if role_session_name is not None and len(role_session_name) > 64:
            raise STSValidationError(
                "1 validation error detected: Value "
                f"'{role_session_name}' "
                "at 'roleSessionName' failed to satisfy constraint: Member must have length less than or "
                f" equal to {MAX_ROLE_NAME_LENGTH}"
            )
        role = self.backend.assume_role(
            region_name=self.region,
            role_session_name=role_session_name,
            role_arn=role_arn,
            policy=policy,
            duration=duration,
            external_id=external_id,
        )
        result = {
            "Credentials": role,
            "AssumedRoleUser": {
                "AssumedRoleId": role.user_id,
                "Arn": role.arn,
            },
            "PackedPolicySize": 6,
        }
        return ActionResult(result)

    def assume_role_with_web_identity(self) -> ActionResult:
        role_session_name = self._get_param("RoleSessionName")
        role_arn = self._get_param("RoleArn")
        policy = self._get_param("Policy")
        duration = self._get_int_param("DurationSeconds", 3600)
        external_id = self._get_param("ExternalId")
        role = self.backend.assume_role_with_web_identity(
            region_name=self.region,
            role_session_name=role_session_name,
            role_arn=role_arn,
            policy=policy,
            duration=duration,
            external_id=external_id,
        )
        result = {
            "Credentials": role,
            "AssumedRoleUser": {
                "AssumedRoleId": (
                    f"{honeypot_profile.stable_id('AROA', role.role_arn, 16)}:"
                    f"{role.session_name}"
                ),
                "Arn": role.arn,
            },
            "PackedPolicySize": 6,
        }
        return ActionResult(result)

    def assume_role_with_saml(self) -> ActionResult:
        role_arn = self._get_param("RoleArn")
        principal_arn = self._get_param("PrincipalArn")
        saml_assertion = self._get_param("SAMLAssertion")
        role = self.backend.assume_role_with_saml(
            role_arn=role_arn,
            principal_arn=principal_arn,
            saml_assertion=saml_assertion,
        )
        result = {
            "Credentials": role,
            "AssumedRoleUser": {
                "AssumedRoleId": role.user_id,
                "Arn": role.arn,
            },
            "PackedPolicySize": 123,
            "Subject": role.user_id,
            "SubjectType": "persistent",
            "Issuer": "http://localhost:3000/",
            "Audience": "https://signin.aws.amazon.com/saml",
            "NameQualifier": "B64EncodedStringOfHashOfIssuerAccountIdAndUserId=",
        }
        return ActionResult(result)

    def get_caller_identity(self) -> ActionResult:
        access_key_id = self.get_access_key()
        user_id, arn, account_id = self.backend.get_caller_identity(
            access_key_id, self.region
        )
        result = {
            "UserId": user_id,
            "Account": account_id,
            "Arn": arn,
        }
        return ActionResult(result)

    def get_access_key_info(self) -> ActionResult:
        access_key_id = self._get_param("AccessKeyId")
        result = self.backend.get_access_key_info(access_key_id)
        return ActionResult(result)
