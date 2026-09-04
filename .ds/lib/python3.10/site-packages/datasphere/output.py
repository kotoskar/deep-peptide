import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Union, TextIO

from google.protobuf.timestamp_pb2 import Timestamp
from tabulate import tabulate

from datasphere.api import jobs_pb2, project_pb2, use_private_api


class Format(Enum):
    TABLE = 'table'
    JSON = 'json'


@dataclass
class ExecutionData:
    job_id: str
    operation_id: str


OutputEntity = Union[jobs_pb2.Job, project_pb2.Project, ExecutionData]


@dataclass
class OutputEntitySpec:
    @dataclass
    class Field:
        value_fetcher: Callable[[OutputEntity], Union[str, Timestamp]]
        name: str  # for json/yaml/whatever output
        title: str  # for table output

    fields: List[Field]


# TODO: delete after renaming `description` -> `desc` in private-api
job_desc_attr = 'description' if use_private_api else 'desc'


job_spec = OutputEntitySpec(
    fields=[
        OutputEntitySpec.Field(lambda job: job.id, 'id', 'ID'),
        OutputEntitySpec.Field(lambda job: job.name, 'name', 'Name'),
        OutputEntitySpec.Field(lambda job: getattr(job, job_desc_attr), 'desc', 'Description'),
        OutputEntitySpec.Field(lambda job: job.created_at, 'created_at', 'Created at'),
        OutputEntitySpec.Field(lambda job: job.finished_at, 'finished_at', 'Finished at'),
        OutputEntitySpec.Field(lambda job: jobs_pb2._JOBSTATUS.values_by_number[job.status].name, 'status', 'Status'),
        OutputEntitySpec.Field(lambda job: job.created_by_id, 'created_by', 'Created by'),
        OutputEntitySpec.Field(lambda job: job.data_cleared, 'data_cleared', 'Data cleared'),
        OutputEntitySpec.Field(lambda job: job.data_expires_at if job.HasField('data_expires_at') else 'Never',
                               'data_expired_at', 'Data expires at'),
    ]
)
project_spec = OutputEntitySpec(
    fields=[
        OutputEntitySpec.Field(lambda prj: prj.id, 'id', 'ID'),
        OutputEntitySpec.Field(lambda prj: prj.name, 'name', 'Name'),
        OutputEntitySpec.Field(lambda prj: prj.community_id, 'community_id', 'Community ID'),
    ]
)
execution_data_spec = OutputEntitySpec(
    fields=[
        OutputEntitySpec.Field(lambda ed: ed.job_id, 'job_id', 'Job ID'),
        OutputEntitySpec.Field(lambda ed: ed.operation_id, 'operation_id', 'Operation ID'),
    ]
)


def output(entities: List[OutputEntity], spec: OutputEntitySpec, target: TextIO, fmt: Union[Format, str]):
    if isinstance(fmt, str):
        fmt = Format(fmt)
    entities_fields: List[List[str]] = []
    for entity in entities:
        fields: List[str] = []
        for field_spec in spec.fields:
            value = field_spec.value_fetcher(entity)
            if isinstance(value, Timestamp):
                value = format_timestamp(value)
            fields.append(value)
        entities_fields.append(fields)
    formatter = {
        Format.TABLE: format_table,
        Format.JSON: format_json,
    }[fmt]
    print(formatter(entities_fields, spec), file=target)


def format_table(entities_fields: List[List[str]], spec: OutputEntitySpec) -> str:
    return tabulate(entities_fields, headers=[f.title for f in spec.fields])


def format_json(entities_fields: List[List[str]], spec: OutputEntitySpec) -> str:
    data = [
        {
            field_spec.name: value
            for field_spec, value in zip(spec.fields, values)
        }
        for values in entities_fields
    ]
    if len(data) == 1:
        data = data[0]
    return json.dumps(data, ensure_ascii=False, indent=4)


def format_timestamp(ts: Optional[Timestamp]) -> str:
    if not ts or (ts.seconds == 0 and ts.nanos == 0):
        return ''
    return ts.ToDatetime().isoformat()
