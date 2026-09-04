from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
import logging
import os
import requests
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
import urllib3

import grpc

from datasphere.api import (
    env,
    ServerEnv,
    iam_endpoint,
    iam_token_service_pb2 as iam_token_service,
    iam_token_service_pb2_grpc as iam_token_service_grpc,
)
from datasphere.version import version

CreateIamTokenRequest = iam_token_service.CreateIamTokenRequest
IamTokenServiceStub = iam_token_service_grpc.IamTokenServiceStub

logger = logging.getLogger(__name__)

# All env vars are for internal usage only.
auth_method_env_var = 'YC_AUTH'
oauth_token_env_var_1 = 'YC_TOKEN'  # As in `yc`
oauth_token_env_var_2 = 'YC_OAUTH_TOKEN'  # More consistent with IAM token env var
# Announced in https://bb.yandexcloud.net/projects/CLOUD/repos/cloud-go/browse/cli/CHANGELOG.md
iam_token_env_var = 'YC_IAM_TOKEN'


class AuthMethod(Enum):
    OAUTH = 'oauth'
    COMPUTE_METADATA = 'md'
    YC_UTILITY = 'yc'


@dataclass
class IssuedIamToken:
    value: str
    expires_after_seconds: int


# We support all auth subjects:
# 1) Usual users
# 2) Federate users
# 3) Service accounts
#
# For 1), user can provide OAuth token, so we generate IAM token with it.
# If OAuth token was not provided, we use `yc` utility as a fallback.
#
# For 2), we delegate auth to `yc` utility, which can handle SSA.
#
# For 3), we either in environment with YC metadata server linked with this SA (Compute Instance, Managed Airflow pod),
# otherwise we delegate auth to `yc` utility.
#
# In case of delegating auth to `yc`, we consider that it configured properly to the corresponding auth subject.
def create_iam_token(oauth_token: Optional[str], yc_profile: Optional[str] = None) -> IssuedIamToken:
    env_token: Optional[str] = os.environ.get(iam_token_env_var)
    if env_token:
        logger.warning('iam token from env var is not refreshable, so it may expire over time')
        return IssuedIamToken(value=env_token, expires_after_seconds=0)

    auth_method_handlers: Dict[AuthMethod, Callable] = {
        AuthMethod.OAUTH: partial(create_iam_token_by_oauth_token, oauth_token=oauth_token),
        AuthMethod.COMPUTE_METADATA: create_iam_token_with_compute_metadata,
        AuthMethod.YC_UTILITY: partial(create_iam_token_with_yc, yc_profile=yc_profile),
    }

    auth_methods = get_auth_methods()
    for method in auth_methods:
        assert method in auth_method_handlers
        handler = auth_method_handlers[method]
        issued_token = handler()
        if issued_token:
            return issued_token

    raise RuntimeError(f'None of authorization methods=[{",".join(m.name for m in auth_methods)}] returned result')


# In priority order
def get_auth_methods():
    auth_methods: Optional[str] = os.environ.get(auth_method_env_var)
    if auth_methods:
        # For internal usages only.
        return [AuthMethod(m) for m in auth_methods.split(',')]
    return [AuthMethod.OAUTH, AuthMethod.COMPUTE_METADATA, AuthMethod.YC_UTILITY]


def create_iam_token_by_oauth_token(oauth_token: Optional[str]) -> Optional[IssuedIamToken]:
    if oauth_token is None:
        return None
    logger.debug('creating iam token using oauth token ...')
    stub = IamTokenServiceStub(grpc.secure_channel(iam_endpoint, grpc.ssl_channel_credentials()))
    req = CreateIamTokenRequest(yandex_passport_oauth_token=oauth_token)
    resp = stub.Create(req)
    return IssuedIamToken(value=resp.iam_token, expires_after_seconds=3600)


metadata_host = os.getenv('YC_METADATA_ADDR', '169.254.169.254')
metadata_server_is_up: Optional[bool] = None  # before first connection check


# This logic duplicates original one from `yandexcloud`, but since we can't use `yandexcloud` because of `protobuf`
# dependencies issues, we moved it here. This logic, including environment variable names, is unlikely to change.
# See https://yandex.cloud/ru/docs/compute/operations/vm-connect/auth-inside-vm#api_3
def create_iam_token_with_compute_metadata() -> Optional[IssuedIamToken]:
    global metadata_server_is_up
    if metadata_server_is_up is False:
        return None

    try:
        resp = requests.get(
            f'http://{metadata_host}/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'},
            timeout=1,
        )
    except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError):
        if metadata_server_is_up:
            logger.error('metadata server was up but now is down')
        metadata_server_is_up = False
        return None
    except Exception as e:
        logger.error('unexpected exception during request to metadata server')
        logger.exception(e)
        metadata_server_is_up = False
        return None

    if metadata_server_is_up is None:
        logger.debug('metadata server found on host %s', metadata_host)
        metadata_server_is_up = True

    logger.debug('iam token is retrieved from metadata server')
    resp = resp.json()
    return IssuedIamToken(value=resp['access_token'], expires_after_seconds=resp['expires_in'])


def create_iam_token_with_yc(yc_profile: Optional[str] = None) -> Optional[IssuedIamToken]:
    logger.debug('creating iam token through `yc` ...')
    cmd = ['yc', 'iam', 'create-token', '--no-user-output']
    if yc_profile:
        cmd += ['--profile', yc_profile]
    try:
        # TODO: capture stderr, process return code
        process = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    except FileNotFoundError:
        logger.warning('`yc` utility was not found. See https://yandex.cloud/cli/ for instructions.')
        return None
    # There may be another output before the token, for example, info about opening the browser.
    # TODO: not sure if token will be last line (update suggestion appears regardless of --no-user-output flag
    token = process.stdout.strip().split('\n')[-1]
    return IssuedIamToken(
        value=token,
        # yc somehow caches IAM token but does not provide info about its expiration time, so let's just
        # re-issue it frequently
        expires_after_seconds=5,
    )


# delta for comparison iam token expiration time
iam_token_expiration_period_epsilon = timedelta(seconds=10)
iam_token_expires_at: Optional[datetime] = None
current_iam_token: Optional[str] = None


def get_md(oauth_token: Optional[str], yc_profile: Optional[str] = None) -> List[Tuple[str, str]]:
    metadata = [("x-client-version", f"datasphere={version}")]

    if env == ServerEnv.DEV:
        return metadata

    global current_iam_token
    global iam_token_expires_at
    now = datetime.now()

    expired = iam_token_expires_at is None or now > iam_token_expires_at - iam_token_expiration_period_epsilon
    if expired:
        issued_token = create_iam_token(oauth_token, yc_profile)
        current_iam_token = issued_token.value
        iam_token_expires_at = now + timedelta(seconds=issued_token.expires_after_seconds)

    assert current_iam_token

    metadata.append(('authorization', f'Bearer {current_iam_token}'))
    return metadata
