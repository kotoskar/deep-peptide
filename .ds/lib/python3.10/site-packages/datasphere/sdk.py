import os
from datetime import datetime
from pathlib import Path

import grpc
from typing import Dict, List, Optional, Union, TextIO
import sys
import logging
import tempfile

from envzy import AutoExplorer

from datasphere.api import jobs_pb2 as jobs, operation_pb2 as operation
from datasphere.client import Client
from datasphere.config import parse_config, check_limits
from datasphere.files import prepare_inputs, prepare_local_modules
from datasphere.fork import fork_params, Overrides
from datasphere.logs import configure_logging
from datasphere.output import output, job_spec, project_spec, Format, ExecutionData, execution_data_spec
from datasphere.pyenv import define_py_env
from datasphere import spark
from datasphere.utils import PythonModule, get_requirements_for_modules
from datasphere.validation import validate_paths
from datasphere.version import version

logger = logging.getLogger(__name__)


class JobWrapper:
    def __init__(
            self,
            job_id: str,
            client: Client,
            op: Optional[operation.Operation] = None,
            execute_call: Optional[grpc.Call] = None,
            job: Optional[jobs.Job] = None,
    ):
        self.job_id = job_id
        self.client = client
        self.op = op
        self.execute_call = execute_call
        self.job = job

    def wait(self):  # TODO: logs destination "None | Pipe"
        if self.op is None or self.execute_call is None:
            raise NotImplementedError('`wait() is not available')
        self.client.read_logs(self.job_id)
        self.client.wait_for_completion(self.op, self.execute_call)

    @property
    def id(self) -> str:
        return self.job_id

    @property
    def operation_id(self) -> str:
        if not self.op:
            raise NotImplementedError('`operation_id` is not available')
        return self.op.id

    @property
    def done(self) -> bool:
        if not self.job:
            raise NotImplementedError('`done` is not available')
        # May be it's better to check operation.done
        return self.job.status in (jobs.JobStatus.SUCCESS, jobs.JobStatus.ERROR, jobs.JobStatus.CANCELLED)


# Underscored methods are used in CLI and not supposed to be used by SDK users.
class SDK:
    def __init__(
            self,
            oauth_token: Optional[str] = None,
            profile: Optional[str] = None,
            log_level: str = 'INFO',
            log_config: Optional[TextIO] = None,
            log_dir: Optional[str] = None,
            _user_agent: Optional[str] = f'datasphere-jobs-sdk-{version}',
    ):
        self.logs_file_path = configure_logging(log_level, log_config, log_dir)
        logger.info('logs file path: %s', self.logs_file_path)
        self.client = Client(oauth_token=oauth_token, yc_profile=profile, user_agent=_user_agent)

    def _execute(self, args):
        cfg = parse_config(args.config)

        py_env = None
        if cfg.env.python:
            logger.debug('defining python env ...')
            py_env = define_py_env(cfg.python_root_modules, cfg.env.python)

        validate_paths(cfg, py_env)

        async_mode = args.async_

        with tempfile.TemporaryDirectory(prefix='datasphere_') as tmpdir:
            logger.debug('using tmp dir `%s` to prepare local files', tmpdir)

            cfg.inputs = prepare_inputs(cfg.inputs, tmpdir)

            if py_env:
                local_modules = [f.get_file() for f in prepare_local_modules(py_env, tmpdir)]
                # Preserve original local modules paths (before archiving).
                sha256_to_display_path = {f.sha256: p for f, p in zip(local_modules, py_env.local_modules_paths)}
            else:
                local_modules = []
                sha256_to_display_path = {}

            check_limits(cfg, local_modules)

            job_params = cfg.get_job_params(py_env, local_modules)
            if py_env:
                logger.debug('resulting python environment:\n%s', py_env.to_string())

            job_id = self.client.create(job_params, cfg, args.project_id, sha256_to_display_path)
            op, execute_call = self.client.execute(job_id)
            logger.debug('operation `%s` executes the job', op.id)

        if not async_mode:
            self.client.read_logs(job_id)

            shutdown_params = job_params.graceful_shutdown_parameters \
                if job_params.HasField("graceful_shutdown_parameters")\
                else None

            self.client.wait_for_completion(op, execute_call, shutdown_params)
        elif args.output != sys.stdout:
            output([ExecutionData(job_id=job_id, operation_id=op.id)], execution_data_spec, args.output, Format.JSON)

    def fork_job(
            self,
            source_job_id: str,
            args: Optional[Dict[str, str]] = None,
            vars: Optional[Dict[str, str]] = None,
            env_vars: Optional[Dict[str, str]] = None,
            name: Optional[str] = None,
            desc: Optional[str] = None,
            docker: Optional[Union[str, dict]] = None,
            working_storage: Optional[dict] = None,
            cloud_instance_types: Optional[List[str]] = None,
    ) -> JobWrapper:
        docker_dict = None
        if docker:
            docker_dict = docker
            if isinstance(docker, str):
                docker_dict = {'docker': docker}
        working_storage_dict = None
        if working_storage:
            working_storage_dict = {'working-storage': working_storage}
        return self.__fork(
            source_job_id,
            Overrides(
                args=args or {},
                vars=vars or {},
                env_vars=env_vars or {},
                name=name,
                desc=desc,
                docker_dict=docker_dict,
                working_storage_dict=working_storage_dict,
                cloud_instance_types=cloud_instance_types or [],
            ),
        )

    def _fork(self, args):
        job_wrapper = self.__fork(args.id, Overrides.from_cli_args(args))

        if not args.async_:
            job_wrapper.wait()
        elif args.output != sys.stdout:
            output(
                [ExecutionData(job_id=job_wrapper.id, operation_id=job_wrapper.operation_id)],
                execution_data_spec, args.output, Format.JSON,
            )

    def __fork(self, source_job_id: str, overrides: Overrides) -> JobWrapper:
        source_job = self.client.get(source_job_id)

        if source_job.HasField('data_expires_at'):
            if source_job.data_cleared:
                raise RuntimeError('fork is unavailable – source job data is cleared')
            data_expires_at = source_job.data_expires_at.ToDatetime()
            logger.warning('source job data will expire after %s, use `set-data-ttl` command to increase '
                           'data TTL to required number of days or to infinity', data_expires_at - datetime.now())

        source_params = source_job.job_parameters

        with tempfile.TemporaryDirectory(prefix='datasphere_') as tmpdir:
            logger.debug('using tmp dir `%s` to prepare local files', tmpdir)
            cfg_overrides = fork_params(overrides, source_params, tmpdir)
            check_limits(cfg_overrides, local_modules=[])
            job_id = self.client.clone(source_job_id, cfg_overrides)
            op, execute_call = self.client.execute(job_id)
            logger.debug('operation `%s` executes the job', op.id)

        return JobWrapper(job_id, self.client, op=op, execute_call=execute_call)

    def _attach(self, args):
        op, execute_call = self.client.execute(args.id)
        self.client.read_logs(args.id, offset=-1)
        self.client.wait_for_completion(op, execute_call)

    def _ls(self, args):
        jobs = self.client.list(args.project_id)
        output(jobs, job_spec, args.output, args.format)

    def get_job(self, id: str) -> JobWrapper:
        job = self.client.get(id)
        return JobWrapper(id, self.client, job=job)

    def _get(self, args):
        job_wrapper = self.get_job(args.id)
        output([job_wrapper.job], job_spec, args.output, args.format)

    def _delete(self, args):
        self.client.delete(args.id)
        logger.info('job deleted')

    def _cancel(self, args):
        self.client.cancel(args.id, args.graceful)

    def _set_data_ttl(self, args):
        days = None  # == 'infinity'
        if args.days.lower() not in ['infinity', 'inf']:
            try:
                days = int(args.days)
            except ValueError:
                raise ValueError('TTL should be either number of days or literal "infinity"')
            if days < 1:
                raise ValueError('TTL days should be positive')
        self.client.set_data_ttl(args.id, days)

    def _ls_projects(self, args):
        projects = self.client.list_projects(args.community_id)
        output(projects, project_spec, args.output, args.format)

    def _get_project(self, args):
        project = self.client.get_project(args.id)
        output([project], project_spec, args.output, args.format)

    def download_job_files(
            self,
            id: str,
            with_logs: bool = False,
            with_diagnostics: bool = False,
            output_dir: Optional[Union[str, Path]] = None
    ):
        job = self.client.get(id)

        if job.status in [jobs.JOB_STATUS_UNSPECIFIED, jobs.CREATING, jobs.EXECUTING, jobs.UPLOADING_OUTPUT]:
            logger.warning(f'job {id} is still running ({job.status}), nothing to download yet.')
            return

        # remains [job.JobStatus.SUCCESS, job.JobStatus.ERROR, job.JobStatus.CANCELLED]
        if job.status != jobs.SUCCESS:
            logger.warning(f'job {id} was completed with error ({job.status}). Not all files can be downloaded.')

        output_files = list(job.output_files)
        if with_logs:
            output_files += job.log_files
        if with_diagnostics:
            output_files += job.diagnostic_files
        self.client.download_files(job.id, output_files, output_dir)

    def _download_job_files(self, args):
        self.download_job_files(
            id=args.id,
            with_logs=args.with_logs,
            with_diagnostics=args.with_diagnostics,
            output_dir=args.output_dir if hasattr(args, 'output_dir') else None
        )

    def _generate_requirements(self, args):
        modules = [PythonModule(path) for path in args.files]
        explorer = AutoExplorer()
        requirements = get_requirements_for_modules(modules, explorer)
        Path(args.output).write_text('\n'.join(requirements))

    def connect_to_spark(self) -> spark.SparkWrapper:
        return spark.connect_to_spark()
