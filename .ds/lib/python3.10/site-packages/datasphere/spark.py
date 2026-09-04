import enum
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.context import SparkContext
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


DATAPROC_CONFIG_PATH_ENV = "JOB_DATAPROC_CONNECTION_CONFIG"
DEFAULT_DATAPROC_CONFIG_PATH = ".spark_connection_config.json"


class ConnectionMode(enum.Enum):
    CONNECT = "SPARK_CONNECT"
    CONTEXT = "SPARK_CONTEXT"
    LOCAL = "SPARK_LOCAL"  # Use only for local testing


@dataclass
class SparkConfig:
    master_node_host: str
    connection_mode: ConnectionMode
    session_params: Dict[str, str]


class SparkWrapper:
    def __init__(self, config: SparkConfig):
        self._config = config
        self._session = self._create_spark_session()

    def _create_spark_session(self) -> 'SparkSession':
        from pyspark.sql import SparkSession

        if self._config.connection_mode == ConnectionMode.CONNECT:
            session_builder = SparkSession.builder.remote(
                f'sc://{self._config.master_node_host}'
            )
        elif self._config.connection_mode == ConnectionMode.CONTEXT:
            session_builder = SparkSession.builder.master('yarn')
        elif self._config.connection_mode == ConnectionMode.LOCAL:
            session_builder = SparkSession.builder.master('local')
        else:
            raise ValueError(f"Unknown mode '{self.connection_mode}'")

        for key, value in self._config.session_params.items():
            session_builder = session_builder.config(key, value)

        return session_builder.getOrCreate()

    @property
    def master_node_host(self) -> str:
        return self._config.master_node_host

    @property
    def connection_mode(self) -> ConnectionMode:
        return self._config.connection_mode

    @property
    def session_params(self) -> Dict[str, str]:
        return self._config.session_params

    @property
    def spark(self) -> 'SparkSession':
        return self._session

    @property
    def sc(self) -> Optional['SparkContext']:
        if self._config.connection_mode == ConnectionMode.CONNECT:
            return None
        return self._session.sparkContext


def parse_spark_config(config: Any) -> SparkConfig:
    if not isinstance(config, dict):
        raise ValueError('Config must be a dict')

    master_node_host = config.get('master_node_host', None)
    if master_node_host is None:
        raise ValueError('Master node host is not specified')
    elif not isinstance(master_node_host, str):
        raise ValueError('Master node host must be a string')

    connection_mode = config.get('connection_mode', None)
    if connection_mode is None:
        raise ValueError('Connection mode is not specified')
    try:
        connection_mode = ConnectionMode(connection_mode)
    except ValueError as error:
        raise ValueError(
            f"Unknown mode '{connection_mode}'. Allowed values: {[mode.name for mode in ConnectionMode]}"
        ) from error

    session_params = config.get('session_params')
    if session_params is None:
        raise ValueError('Spark session params are not specified')
    elif not (
        isinstance(session_params, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in session_params.items())
    ):
        raise ValueError('Spark session params must be a dict of strings')

    return SparkConfig(
        master_node_host=master_node_host,
        connection_mode=ConnectionMode(connection_mode),
        session_params=session_params,
    )


def load_spark_config(path: Path) -> SparkConfig:
    if not os.path.exists(path) and not os.path.isfile(path):
        raise FileNotFoundError(
            f"Config path '{path}' is not found or is not a regular file"
        )

    logger.debug(f'Found spark config at: {path}')

    with open(path) as file:
        config = json.load(file)

    return parse_spark_config(config)


def connect_to_spark() -> SparkWrapper:
    config_path = os.getenv(DATAPROC_CONFIG_PATH_ENV, DEFAULT_DATAPROC_CONFIG_PATH)
    config = load_spark_config(config_path)
    return SparkWrapper(config)
