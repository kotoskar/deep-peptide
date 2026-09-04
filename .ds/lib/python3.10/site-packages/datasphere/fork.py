from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from datasphere.api import jobs_pb2 as jobs
from datasphere.config import Config, Environment, VariablePath, parse_docker_image, parse_working_storage
from datasphere.files import prepare_inputs
from datasphere.validation import validate_paths


@dataclass
class Overrides:
    args: Dict[str, str] = field(default_factory=dict)
    vars: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    name: Optional[str] = None
    desc: Optional[str] = None
    docker_dict: Optional[dict] = None
    working_storage_dict: Optional[dict] = None
    cloud_instance_types: List[str] = field(default_factory=list)

    @staticmethod
    def from_cli_args(args) -> 'Overrides':
        return Overrides(
            args=parse_name_value_list(args.arg),
            vars=parse_name_value_list(args.var),
            env_vars=parse_name_value_list(args.env_var),
            name=args.name,
            desc=args.desc,
            docker_dict=get_docker_dict_from_cli_args(args),
            working_storage_dict=get_working_storage_dict_from_cli_args(args),
            cloud_instance_types=args.cloud_instance_type or [],
        )


def fork_params(overrides: Overrides, source_params: jobs.JobParameters, tmpdir) -> Config:
    var_to_old_input = {
        f.desc.var: VariablePath(path=f.desc.path, var=f.desc.var)
        for f in source_params.input_files
        if f.desc.var
    }
    var_to_old_output = {
        f.var: VariablePath(path=f.path, var=f.var)
        for f in source_params.output_files
        if f.var
    }

    new_inputs = []
    new_outputs = []
    new_vars = set()
    for var, path in overrides.vars.items():
        var_path = VariablePath(var=var, path=path)
        new_vars.add(var)
        if var in var_to_old_input:
            new_inputs.append(var_path)
        elif var in var_to_old_output:
            new_outputs.append(var_path)
        else:
            raise ValueError(f'No variable path in source job: {var}')

    source_args = {arg.name for arg in source_params.arguments}
    new_args = {}
    for name, val in overrides.args.items():
        if name not in source_args:
            raise ValueError(f"Argument with name '{name}' is not present in source job")
        new_args[name] = val

    # Environment override details:
    # - Docker image from source job can only be changed, but not reset, so it's ok to pass `None` as docker image
    #   even if source job has some docker image.
    # - Python env cannot be modified.
    # - Environment variables, if set, substitute source job ones.
    env = Environment(
        vars=overrides.env_vars,
        docker_image=parse_docker_image(overrides.docker_dict or {}),
        python=None,
    )

    working_storage = parse_working_storage(overrides.working_storage_dict or {})

    # All parameters which cannot be overridden are left with their default values.
    cfg = Config(
        cmd='',
        args=new_args,
        inputs=new_inputs,
        outputs=new_outputs,
        s3_mounts=[],
        datasets=[],
        env=env,
        cloud_instance_types=overrides.cloud_instance_types,
        attach_project_disk=False,
        content='',
        working_storage=working_storage,
        name=overrides.name,
        desc=overrides.desc,
    )

    # We need to check all paths on overlaps – overridden and source ones.
    old_inputs = [old_input for var, old_input in var_to_old_input.items() if var not in new_vars]
    old_outputs = [old_output for var, old_output in var_to_old_output.items() if var not in new_vars]
    cfg_joint_io = replace(cfg, inputs=new_inputs + old_inputs, outputs=new_outputs + old_outputs)
    validate_paths(cfg_joint_io, py_env=None)

    cfg.inputs = prepare_inputs(cfg.inputs, tmpdir)

    return cfg


def get_docker_dict_from_cli_args(args) -> Optional[dict]:
    if args.docker_image_id or args.docker_image_url:
        if args.docker_image_id and args.docker_image_url:
            raise ValueError('For docker image, specify either ID or URL')
        if args.docker_image_id:
            return {'docker': args.docker_image_id}
        else:
            data = {'image': args.docker_image_url}
            if args.docker_image_username:
                data['username'] = args.docker_image_username
            if args.docker_image_password_secret_id:
                data['password'] = {'secret-id': args.docker_image_password_secret_id}
            return {'docker': data}
    return None


def get_working_storage_dict_from_cli_args(args) -> Optional[dict]:
    if args.working_storage_type or args.working_storage_size:
        d = {}
        if args.working_storage_type:
            d['type'] = args.working_storage_type
        if args.working_storage_size:
            d['size'] = args.working_storage_size
        return {'working-storage': d}
    return None


def parse_name_value(s: str) -> Tuple[str, str]:
    # split by first `=` since it cannot be in path var or env var
    parts = s.split('=', 1)
    if len(parts) != 2:
        raise ValueError(f'Invalid name-value pair: {s}')
    return parts[0], parts[1]


def parse_name_value_list(lst: Optional[List[str]]) -> dict:
    return {
        name: value for name, value in
        [parse_name_value(arg) for arg in lst or []]
    }
