from enum import Enum
import os
from typing import Optional


class ServerEnv(Enum):
    DEV = 'DEV'
    PREPROD_NO_GW = 'PREPROD_NO_GW'  # Go directly to unified-room and lobby host bypassing gateway
    PREPROD = 'PREPROD'
    PROD = 'PROD'


env_str: Optional[str] = os.getenv('SERVER_ENV', None)
env: ServerEnv = ServerEnv(env_str.upper()) if env_str else ServerEnv.PROD
use_private_api = env in (ServerEnv.DEV, ServerEnv.PREPROD_NO_GW)

project_endpoint, job_endpoint, op_endpoint, iam_endpoint, ui_endpoint = {
    ServerEnv.DEV: ('localhost:9098', 'localhost:9098', 'localhost:9098', 'iam.api.cloud-preprod.yandex.net:443', ''),
    ServerEnv.PREPROD_NO_GW: (
        'lobby.datasphere.cloud-preprod.yandex.net:9090',
        'unified-room.datasphere.cloud-preprod.yandex.net:9090',
        'lobby.datasphere.cloud-preprod.yandex.net:9090',
        'iam.api.cloud-preprod.yandex.net:443',
        'datasphere-preprod.yandex.cloud',
    ),
    ServerEnv.PREPROD: (
        'datasphere.api.cloud-preprod.yandex.net:443',
        'datasphere.api.cloud-preprod.yandex.net:443',
        'operation.api.cloud-preprod.yandex.net:443',
        'iam.api.cloud-preprod.yandex.net:443',
        'datasphere-preprod.yandex.cloud',
    ),
    ServerEnv.PROD: (
        'datasphere.api.cloud.yandex.net:443',
        'datasphere.api.cloud.yandex.net:443',
        'operation.api.cloud.yandex.net:443',
        'iam.api.cloud.yandex.net:443',
        'datasphere.yandex.cloud',
    ),
}[env]

if use_private_api:
    from ..yandex.cloud.priv.datasphere.v2.jobs import (
        jobs_pb2,
        jobs_pb2_grpc,
        project_job_service_pb2,
        project_job_service_pb2_grpc
    )

    from ..yandex.cloud.priv.datasphere.v1 import (
        operation_service_pb2,
        operation_service_pb2_grpc
    )

    from ..yandex.cloud.priv.datasphere.v2 import (
        project_pb2,
        project_service_pb2,
        project_service_pb2_grpc
    )

    from ..yandex.cloud.priv.operation import (
        operation_pb2
    )

    from ..yandex.cloud.priv.iam.v1 import (
        iam_token_service_pb2,
        iam_token_service_pb2_grpc
    )
else:
    try:
        from yandex.cloud.datasphere.v2.jobs import (
            jobs_pb2,
            jobs_pb2_grpc,
            project_job_service_pb2,
            project_job_service_pb2_grpc
        )
    except ImportError:
        from ..yandex.cloud.datasphere.v2.jobs import (
            jobs_pb2,
            jobs_pb2_grpc,
            project_job_service_pb2,
            project_job_service_pb2_grpc
        )

    try:
        from yandex.cloud.operation import (
            operation_service_pb2,
            operation_service_pb2_grpc
        )
    except ImportError:
        from ..yandex.cloud.operation import (
            operation_service_pb2,
            operation_service_pb2_grpc
        )

    try:
        from yandex.cloud.datasphere.v2 import (
            project_pb2,
            project_service_pb2,
            project_service_pb2_grpc
        )
    except ImportError:
        from ..yandex.cloud.datasphere.v2 import (
            project_pb2,
            project_service_pb2,
            project_service_pb2_grpc
        )

    try:
        from yandex.cloud.operation import (
            operation_pb2
        )
    except ImportError:
        from ..yandex.cloud.operation import (
            operation_pb2
        )

    try:
        from yandex.cloud.iam.v1 import (
            iam_token_service_pb2,
            iam_token_service_pb2_grpc
        )
    except ImportError:
        from ..yandex.cloud.iam.v1 import (
            iam_token_service_pb2,
            iam_token_service_pb2_grpc
        )
