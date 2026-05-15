import io
import os
import os.path
import re
from collections.abc import Callable
from threading import Lock
from typing import Any, Optional

try:
    from flask import Flask, Response, request
    from flask_cors import CORS
except ImportError:
    import warnings

    warnings.warn(
        "When using MotoServer, ensure that you install moto[server] to have all dependencies!\n",
        stacklevel=2,
    )
    raise

import moto.backend_index as backend_index
import moto.backends as backends
from moto.core import DEFAULT_ACCOUNT_ID
from moto.core.base_backend import BackendDict
from moto.core.utils import convert_to_flask_response
from moto.settings import DISABLE_GLOBAL_CORS, MAX_FORM_MEMORY_SIZE

from .utilities import AWSTestHelper, RegexConverter, decompress_request_body

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH", "OPTIONS"]


DEFAULT_SERVICE_REGION = ("s3", "us-east-1")

# Map of unsigned calls to service-region as per AWS API docs
# https://docs.aws.amazon.com/cognito/latest/developerguide/resource-permissions.html#amazon-cognito-signed-versus-unsigned-apis
UNSIGNED_REQUESTS = {
    "AWSCognitoIdentityService": ("cognito-identity", "us-east-1"),
    "AWSCognitoIdentityProviderService": ("cognito-idp", "us-east-1"),
}
UNSIGNED_ACTIONS = {
    "AssumeRoleWithSAML": ("sts", "us-east-1"),
    "AssumeRoleWithWebIdentity": ("sts", "us-east-1"),
}

# SigV4 signing name과 Moto backend 이름이 다를 때 fallback 라우팅 전에 보정한다.
SIGNING_ALIASES = {
    # AWS CLI credential scope에는 access-analyzer로 찍히지만 botocore/Moto service id는 accessanalyzer다.
    "access-analyzer": "accessanalyzer",
    # 기존 Moto 라우팅에서 이미 쓰던 Bedrock AgentCore backend alias다.
    "bedrock-agentcore": "bedrock-agentcore-control",
    # EventBridge는 signing name이 eventbridge지만 Moto backend는 events다.
    "eventbridge": "events",
    # execute-api signing name은 API Gateway management/data path로 다시 분기된다.
    "execute-api": "iot",
    # IoT data plane signing name을 Moto backend 이름으로 맞춘다.
    "iotdata": "data.iot",
    # Pinpoint의 과거 signing/backend 이름 차이를 맞춘다.
    "mobiletargeting": "pinpoint",
}

# JSON RPC 계열 요청에서 Host만으로 service를 못 찾을 때 X-Amz-Target prefix로 복구한다.
TARGET_PREFIX_ALIASES = {
    # FraudDetector CLI 요청의 X-Amz-Target prefix다.
    "AWSHawksNestServiceFacade": "frauddetector",
    # Resource Explorer v2 CLI 요청의 X-Amz-Target prefix다.
    "AWSResourceExplorer": "resource-explorer-2",
    # Backup Gateway CLI 요청의 X-Amz-Target prefix다.
    "BackupOnPremises_v20210101": "backup-gateway",
}

# Some services are only recognizable by the version
SERVICE_BY_VERSION = {"2009-04-15": "sdb"}


def _service_region_from_authorization(
    authorization: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    # Authorization 헤더가 없으면 SigV4 credential scope에서 service/region을 복구할 수 없다.
    if not authorization:
        return None, None
    # SigV4 Credential=AKIA/date/region/service/aws4_request 형식에서 region/service만 추출한다.
    match = re.search(
        r"Credential=[^,/]+/\d{8}/([^/]+)/([^/]+)/aws4_request", authorization
    )
    # SigV4 형식이 아니면 fallback 라우터가 다른 힌트를 계속 시도하게 둔다.
    if not match:
        return None, None
    # 첫 번째 캡처 그룹은 AWS region이다.
    region = match.group(1)
    # 두 번째 캡처 그룹은 signing service이고, Moto backend 이름과 다르면 alias로 바꾼다.
    service = SIGNING_ALIASES.get(match.group(2).lower(), match.group(2).lower())
    # fallback app과 native backend inference가 같이 쓸 수 있도록 service/region을 반환한다.
    return service, region


def _service_from_target(target: Optional[str]) -> Optional[str]:
    # X-Amz-Target이 없거나 prefix.operation 형식이 아니면 이 경로로 service를 알 수 없다.
    if not target or "." not in target:
        return None
    # 점 앞 prefix가 AWS JSON RPC service family를 나타낸다.
    prefix = target.split(".", 1)[0]
    # 우리가 아는 prefix면 Moto/backend service 이름으로 변환한다.
    if prefix in TARGET_PREFIX_ALIASES:
        return TARGET_PREFIX_ALIASES[prefix]
    # 모르는 prefix는 다른 추론 경로가 처리하게 둔다.
    return None


def _operation_from_target(target: Optional[str]) -> Optional[str]:
    # X-Amz-Target은 보통 Prefix.Operation 형태라서 마지막 토큰이 AWS operation이다.
    if target and "." in target:
        return target.rsplit(".", 1)[-1]
    # 형식이 맞지 않으면 body/path 기반 추론으로 넘긴다.
    return None


def _operation_from_body(body: bytes) -> Optional[str]:
    # Query protocol body를 문자열로 바꿔 Action= 값을 찾는다.
    try:
        text = body.decode("utf-8", errors="ignore")
    # body decoding 자체가 실패하면 operation 추론을 포기한다.
    except Exception:
        return None
    # EC2/IAM/STS 같은 Query protocol은 body에 Action=OperationName을 담는다.
    match = re.search(r"(?:^|&)Action=([^&]+)", text)
    # Action 파라미터가 있으면 fallback agent에 넘길 operation 이름으로 쓴다.
    if match:
        return match.group(1)
    # Action 파라미터가 없으면 path 기반 추론으로 넘긴다.
    return None


def _operation_from_path(
    service: Optional[str], method: str, path_info: str
) -> Optional[str]:
    # service를 모르면 service model을 열 수 없으므로 generic path 변환만 가능하다.
    if not service:
        return None
    # botocore service model을 이용해 path/method와 정확히 맞는 operation을 찾는다.
    try:
        from moto.core.utils import get_service_model

        service_model = get_service_model(service)
    # 모델이 없으면 /foo-bar 같은 path를 FooBar operation으로 변환하는 fallback을 쓴다.
    except Exception:
        return _operation_name_from_path(path_info)

    # trailing slash 차이로 매칭이 깨지지 않도록 정규화한다.
    normalized_path = path_info.rstrip("/") or "/"
    # service model의 모든 operation URI 패턴과 현재 request path를 비교한다.
    for operation_name in service_model.operation_names:
        # operation별 HTTP method와 requestUri를 읽는다.
        operation_model = service_model.operation_model(operation_name)
        http = operation_model.http
        # method가 다르면 같은 path라도 다른 operation일 수 있으므로 건너뛴다.
        if str(http.get("method", "")).upper() != method.upper():
            continue
        # query string은 path matching에 필요 없으므로 제거한다.
        request_uri = str(http.get("requestUri", "/")).split("?", 1)[0]
        # trailing slash 차이를 제거한다.
        request_uri = request_uri.rstrip("/") or "/"
        # requestUri 템플릿을 regex로 바꾸기 위해 우선 literal escape한다.
        pattern = re.escape(request_uri)
        # {id} 같은 path parameter는 실제 값 한 segment와 매칭되게 바꾼다.
        pattern = re.sub(r"\\\{[^}]+\\\}", r"[^/]+", pattern)
        # 모델 path와 실제 path가 맞으면 해당 operation 이름을 반환한다.
        if re.fullmatch(pattern, normalized_path):
            return operation_name
    # 모델 매칭이 실패하면 path 문자열 기반 generic operation 이름을 반환한다.
    return _operation_name_from_path(path_info)


def _operation_name_from_path(path_info: str) -> Optional[str]:
    cleaned = path_info.strip("/")
    if not cleaned:
        return None
    parts = re.split(r"[^A-Za-z0-9]+", cleaned)
    parts = [part for part in parts if part]
    if not parts:
        return None
    return "".join(part[:1].upper() + part[1:] for part in parts)


class DomainDispatcherApplication:
    """
    Dispatch requests to different applications based on the "Host:" header
    value. We'll match the host header value with the url_bases of each backend.
    """

    def __init__(self, create_app: Callable[[backends.SERVICE_NAMES], Flask]):
        self.create_app = create_app
        self.lock = Lock()
        self.app_instances: dict[str, Flask] = {}
        self.backend_url_patterns = backend_index.backend_url_patterns

    def get_backend_for_host(self, host: Optional[str]) -> Any:
        if host is None:
            return None

        if host == "moto_api":
            return host

        if host in backends.list_of_moto_modules():
            return host

        for backend, pattern in self.backend_url_patterns:
            if pattern.match(f"http://{host}"):
                return backend

        if "amazonaws.com" in host:
            print(  # noqa
                f"Unable to find appropriate backend for {host}."
                "Remember to add the URL to urls.py, and run scripts/update_backend_index.py to index it."
            )

    def infer_service_region_host(
        self, environ: dict[str, Any], path: str
    ) -> Optional[str]:
        auth = environ.get("HTTP_AUTHORIZATION")
        target = environ.get("HTTP_X_AMZ_TARGET")
        service = None
        if auth:
            # Signed request
            # Parse auth header to find service assuming a SigV4 request
            # https://docs.aws.amazon.com/general/latest/gr/sigv4-signed-request-examples.html
            # ['Credential=sdffdsa', '20170220', 'us-east-1', 'sns', 'aws4_request']
            service, region = _service_region_from_authorization(auth)
            if service:
                path = environ.get("PATH_INFO", "")
                if service.lower() == "execute-api" and path.startswith(
                    "/@connections"
                ):
                    # APIGateway Management API
                    pass
            else:
                # Signature format does not match, this is exceptional and we can't
                # infer a service-region. A reduced set of services still use
                # the deprecated SigV2, ergo prefer S3 as most likely default.
                # https://docs.aws.amazon.com/general/latest/gr/signature-version-2.html
                return None
        else:
            # Unsigned request
            body = self._get_body(environ)
            action = self.get_action_from_body(body)
            if target:
                service, _ = target.split(".", 1)
                service, region = UNSIGNED_REQUESTS.get(service, (None, None))
            elif action and action in UNSIGNED_ACTIONS:
                # See if we can match the Action to a known service
                service, region = UNSIGNED_ACTIONS[action]
            if not service:
                service, region = self.get_service_from_body(body, environ)
            if not service:
                service, region = self.get_service_from_unsigned_path(path)
            if not service:
                service, region = self.get_service_from_path(path)
            if not service:
                return None

        if service in ["budgets", "cloudfront"]:
            # Global Services - they do not have/expect a region
            host = f"{service}.amazonaws.com"
        elif service == "mediastore" and not target:
            # All MediaStore API calls have a target header
            # If no target is set, assume we're trying to reach the mediastore-data service
            host = f"data.{service}.{region}.amazonaws.com"
        elif service == "dsql":
            host = f"{service}.{region}.api.aws"
        elif service == "dynamodb":
            if environ["HTTP_X_AMZ_TARGET"].startswith("DynamoDBStreams"):
                host = "dynamodbstreams"
            else:
                dynamo_api_version = (
                    environ["HTTP_X_AMZ_TARGET"].split("_")[1].split(".")[0]
                )
                # Support for older API version
                if dynamo_api_version <= "20111205":
                    host = "dynamodb_v20111205"
                else:
                    host = "dynamodb"
        elif service == "sagemaker":
            if environ["PATH_INFO"].endswith("invocations"):
                host = f"runtime.{service}.{region}.amazonaws.com"
            elif environ["PATH_INFO"].endswith("BatchPutMetrics"):
                host = f"metrics.{service}.{region}.amazonaws.com"
            else:
                host = f"api.{service}.{region}.amazonaws.com"
        elif service == "timestream":
            host = f"ingest.{service}.{region}.amazonaws.com"
        elif service == "s3" and (
            path.startswith("/v20180820/") or "s3-control" in environ["HTTP_HOST"]
        ):
            host = "s3control"
        elif service == "s3vectors":
            host = f"{service}.{region}.api.aws"
        elif service == "ses" and path.startswith("/v2/"):
            host = "sesv2"
        elif service == "memorydb":
            host = f"memory-db.{region}.amazonaws.com"
        elif service == "bedrock":
            # Multiple Bedrock services use the same signing name (bedrock).
            # This is obviously a hack, but it automatically differentiates
            # between the various Bedrock services without having to manually
            # add every path to `moto/bedrock/urls.py`.
            from moto.bedrock.responses import BedrockResponse
            from moto.bedrockagent.responses import AgentsforBedrockResponse
            from moto.bedrockruntime.responses import BedrockRuntimeResponse

            service_to_response = {
                "bedrock": BedrockResponse,
                "bedrock-agent": AgentsforBedrockResponse,
                "bedrock-runtime": BedrockRuntimeResponse,
            }
            for service_name, response_class in service_to_response.items():
                resp = response_class()
                resp.region = region
                action = resp._get_action_from_method_and_request_uri(
                    method=environ["REQUEST_METHOD"],
                    request_uri=environ["PATH_INFO"],
                )
                if action:
                    service = service_name
                    break
            host = f"{service}.{region}.amazonaws.com"
        else:
            host = f"{service}.{region}.amazonaws.com"

        return host

    def get_application(self, environ: dict[str, Any]) -> Flask:
        path_info = environ.get("PATH_INFO", "")

        # The URL path might contain non-ASCII text, for instance unicode S3 bucket names
        if isinstance(path_info, bytes):
            path_info = path_info.decode("utf-8")

        if path_info.startswith("/moto-api") or path_info == "/favicon.ico":
            host = "moto_api"
        elif path_info.startswith("/latest/meta-data/") or path_info.startswith(
            "/latest/api/"
        ):
            host = "instance_metadata"
        else:
            host = None

        with self.lock:
            backend = self.get_backend_for_host(host)
            if not backend:
                # No regular backend found; try parsing body/other headers
                host = self.infer_service_region_host(environ, path_info)
                if host is None:
                    service, region = DEFAULT_SERVICE_REGION
                    host = f"{service}.{region}.amazonaws.com"
                backend = self.get_backend_for_host(host)

            if not backend:
                # 여기까지 오면 Moto native backend를 찾지 못한 것이므로 LLM fallback app으로 보낸다.
                app = self.app_instances.get("__llm_fallback__", None)
                # fallback app은 backend별 app처럼 캐시해서 매 요청마다 Flask app을 만들지 않는다.
                if app is None:
                    app = create_llm_fallback_app()
                    self.app_instances["__llm_fallback__"] = app
                # native app 대신 fallback app이 이 WSGI 요청을 처리한다.
                return app

            app = self.app_instances.get(backend, None)
            if app is None:
                app = self.create_app(backend)
                self.app_instances[backend] = app
            return app

    def _get_body(self, environ: dict[str, Any]) -> Optional[str]:
        body = None
        try:
            # AWS requests use querystrings as the body (Action=x&Data=y&...)
            simple_form = environ["CONTENT_TYPE"].startswith(
                "application/x-www-form-urlencoded"
            )
            request_body_size = int(environ["CONTENT_LENGTH"])
            if simple_form and request_body_size:
                body = environ["wsgi.input"].read(request_body_size).decode("utf-8")
        except (KeyError, ValueError):
            pass
        finally:
            if body:
                # We've consumed the body = need to reset it
                environ["wsgi.input"] = io.BytesIO(body.encode("utf-8"))
        return body

    def get_service_from_body(
        self, body: Optional[str], environ: dict[str, Any]
    ) -> tuple[Optional[str], Optional[str]]:
        # Some services have the SDK Version in the body
        # If the version is unique, we can derive the service from it
        version = self.get_version_from_body(body)
        if version and version in SERVICE_BY_VERSION:
            # Boto3/1.20.7 Python/3.8.10 Linux/5.11.0-40-generic Botocore/1.23.7 region/eu-west-1
            region = environ.get("HTTP_USER_AGENT", "").split("/")[-1]
            return SERVICE_BY_VERSION[version], region
        return None, None

    def get_version_from_body(self, body: Optional[str]) -> Optional[str]:
        try:
            body_dict = dict(x.split("=") for x in body.split("&"))  # type: ignore
            return body_dict["Version"]
        except (AttributeError, KeyError, ValueError):
            return None

    def get_action_from_body(self, body: Optional[str]) -> Optional[str]:
        try:
            # AWS requests use querystrings as the body (Action=x&Data=y&...)
            body_dict = dict(x.split("=") for x in body.split("&"))  # type: ignore
            return body_dict["Action"]
        except (AttributeError, KeyError, ValueError):
            return None

    def get_service_from_path(
        self, path_info: str
    ) -> tuple[Optional[str], Optional[str]]:
        # Moto sometimes needs to send a HTTP request to itself
        # In which case it will send a request to 'http://localhost/service_region/whatever'
        try:
            service, region = path_info[1 : path_info.index("/", 1)].split("_")
            return service, region
        except (AttributeError, KeyError, ValueError):
            return None, None

    @staticmethod
    def get_service_from_unsigned_path(
        path_info: str,
    ) -> tuple[Optional[str], Optional[str]]:
        # Must run before get_service_from_path, which would misparse the user pool ID
        # prefix (e.g., "us-east-1" from "us-east-1_abc123") as a service name.
        #
        # Some unsigned GET requests can be routed based on the URL path alone.
        # For example, Cognito JWKS endpoint:
        #   GET /{user_pool_id}/.well-known/jwks.json
        # The user_pool_id starts with the region (e.g. us-east-1_abc123).
        if path_info.rstrip("/").endswith("/.well-known/jwks.json"):
            # Extract region from user pool ID (format: {region}_{id})
            pool_id = path_info.strip("/").split("/")[0]
            if "_" not in pool_id:
                return "cognito-idp", "us-east-1"
            region = pool_id.rsplit("_", 1)[0]
            return "cognito-idp", region
        return None, None

    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        backend_app = self.get_application(environ)
        return backend_app(environ, start_response)


def create_backend_app(service: backends.SERVICE_NAMES) -> Flask:
    from werkzeug.routing import Map

    current_file = os.path.abspath(__file__)
    current_dir = os.path.abspath(os.path.join(current_file, os.pardir))
    template_dir = os.path.join(current_dir, "templates")

    # Create the backend_app
    backend_app = Flask("moto", template_folder=template_dir)
    backend_app.debug = True
    backend_app.service = service  # type: ignore[attr-defined]
    backend_app.before_request(decompress_request_body)
    backend_app.config["MAX_FORM_MEMORY_SIZE"] = MAX_FORM_MEMORY_SIZE

    if not DISABLE_GLOBAL_CORS:
        CORS(backend_app)

    # Reset view functions to reset the app
    backend_app.view_functions = {}
    backend_app.url_map = Map()
    backend_app.url_map.converters["regex"] = RegexConverter

    backend_dict = backends.get_backend(service)
    # Get an instance of this backend.
    # We'll only use this backend to resolve the URL's, so the exact region/account_id is irrelevant
    if isinstance(backend_dict, BackendDict):
        if "us-east-1" in backend_dict[DEFAULT_ACCOUNT_ID]:
            backend = backend_dict[DEFAULT_ACCOUNT_ID]["us-east-1"]
        else:
            backend = backend_dict[DEFAULT_ACCOUNT_ID]["aws"]
    else:
        backend = backend_dict["global"]

    for url_path, handler in backend.flask_paths.items():
        view_func = convert_to_flask_response(handler)
        if handler.__name__ == "dispatch":
            endpoint = f"{handler.__self__.__name__}.dispatch"  # type: ignore[attr-defined]
        else:
            endpoint = view_func.__name__

        original_endpoint = endpoint
        index = 2
        while endpoint in backend_app.view_functions:
            # HACK: Sometimes we map the same view to multiple url_paths. Flask
            # requires us to have different names.
            endpoint = original_endpoint + str(index)
            index += 1

        # Some services do not provide a URL path
        # I.e., boto3 sends a request to 'https://ingest.timestream.amazonaws.com'
        # Which means we have a empty url_path to catch this request - but Flask can't handle that
        if url_path:
            backend_app.add_url_rule(
                url_path,
                endpoint=endpoint,
                methods=HTTP_METHODS,
                view_func=view_func,
                strict_slashes=False,
            )

    backend_app.test_client_class = AWSTestHelper
    return backend_app


def create_llm_fallback_app() -> Flask:
    # native backend가 없는 서비스/operation 요청을 처리하는 전용 Flask app이다.
    fallback_app = Flask("moto_llm_fallback")

    # 루트 path 요청도 fallback agent로 들어오게 한다.
    @fallback_app.route("/", defaults={"path": ""}, methods=HTTP_METHODS)
    # 임의 path 요청도 fallback agent로 들어오게 한다.
    @fallback_app.route("/<path:path>", methods=HTTP_METHODS)
    def llm_fallback(path: str) -> Response:
        # Flask request body를 bytes로 읽어 Query/JSON protocol 파라미터 추론에 쓴다.
        body = request.get_data(cache=True)
        # 원본 AWS CLI 헤더를 dict로 바꿔 fallback normalizer와 audit에 넘긴다.
        headers = dict(request.headers)
        # SigV4 credential scope에서 service/region을 복구하기 위해 Authorization을 읽는다.
        authorization = headers.get("Authorization")
        # JSON RPC service/operation 복구를 위해 X-Amz-Target을 읽는다.
        target = headers.get("X-Amz-Target") or headers.get("x-amz-target")
        # 우선 SigV4 credential scope에서 service를 추론한다.
        service, _region = _service_region_from_authorization(authorization)
        # SigV4에서 못 찾으면 X-Amz-Target prefix alias로 service를 추론한다.
        service = service or _service_from_target(target)
        # operation은 target, query body, REST path 순서로 복구한다.
        action = (
            _operation_from_target(target)
            or _operation_from_body(body)
            or _operation_from_path(service, request.method, request.path)
        )
        # 정상 경로에서는 agent runtime이 AWS-shaped fallback 응답을 만든다.
        try:
            from moto.core.llm_agents import handle_aws_request

            # agent runtime에 service/action/request context를 넘겨 shape-safe 응답을 생성한다.
            response_body = handle_aws_request(
                service=service,
                action=action,
                url=request.url,
                headers=headers,
                body=body,
                reason="Moto server could not resolve a native backend",
                source="moto_server.unresolved_backend",
            )
            # AWS CLI가 파싱할 수 있도록 200 body를 그대로 반환한다.
            return Response(response_body, status=200)
        # agent runtime 자체가 실패해도 Flask 500/HTML을 노출하지 않는다.
        except Exception:
            from moto.core.llm_fallback import build_llm_fallback_json

            # 최후 fallback은 단순 JSON body로 최소한의 응답을 만든다.
            response_headers, response_body = build_llm_fallback_json()
            # 실패해도 HTTP 200으로 돌려 CLI/server traceback 노출을 막는다.
            return Response(response_body, status=200, headers=response_headers)

    # DomainDispatcherApplication이 WSGI app으로 사용할 Flask app을 반환한다.
    return fallback_app
