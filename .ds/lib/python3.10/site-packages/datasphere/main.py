import argparse
import sys
import grpc
import logging
import os
from pathlib import Path

from datasphere.auth import oauth_token_env_var_1, oauth_token_env_var_2
from datasphere.client import OperationError, ProgramError
from datasphere.output import Format
from datasphere.utils import check_package_version, print_logs_files
from datasphere.version import version
from datasphere.sdk import SDK


def execute(args, sdk):
    sdk._execute(args)


def fork(args, sdk):
    sdk._fork(args)


def attach(args, sdk):
    sdk._attach(args)


def ls(args, sdk):
    sdk._ls(args)


def get(args, sdk):
    sdk._get(args)


def delete(args, sdk):
    sdk._delete(args)


def cancel(args, sdk):
    sdk._cancel(args)


def set_data_ttl(args, sdk):
    sdk._set_data_ttl(args)


def download_job_files(args, sdk):
    sdk._download_job_files(args)


def ls_projects(args, sdk):
    sdk._ls_projects(args)


def get_project(args, sdk):
    sdk._get_project(args)


def show_version(_, __):
    print(version)


def show_changelog(_, __):
    print(Path(__file__).with_name('changelog.md').read_text())


def generate_requirements(args, sdk):
    sdk._generate_requirements(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser_datasphere = argparse.ArgumentParser(prog='datasphere')
    parser_datasphere.add_argument(
        '-t', '--token',
        default=os.environ.get(oauth_token_env_var_1) or os.environ.get(oauth_token_env_var_2),
        help='YC OAuth token, see https://cloud.yandex.com/docs/iam/concepts/authorization/oauth-token'
    )
    parser_datasphere.add_argument(
        '-l', '--log-level', default=logging.INFO,
        choices=[logging.getLevelName(level) for level in (logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG)],
        help='Logging level',
    )
    parser_datasphere.add_argument(
        '--log-config', type=argparse.FileType('r'), help='Custom logging config'
    )
    parser_datasphere.add_argument('--log-dir', help='Logs directory (temporary directory by default)')
    parser_datasphere.add_argument('--profile', help='`yc` utility profile')
    parser_datasphere.add_argument('--user-agent', help=argparse.SUPPRESS, default=f'datasphere-jobs-cli-{version}')
    subparsers_datasphere = parser_datasphere.add_subparsers(required=True)

    parser_version = subparsers_datasphere.add_parser('version', help='Show version')
    parser_version.set_defaults(func=show_version)

    parser_changelog = subparsers_datasphere.add_parser('changelog', help='Show changelog')
    parser_changelog.set_defaults(func=show_changelog)

    parser_project = subparsers_datasphere.add_parser('project')
    subparsers_project = parser_project.add_subparsers(required=True)

    parser_generate_req = subparsers_datasphere.add_parser(
        'generate-requirements',
        help='Generate requirements for specified root module(s)'
    )
    parser_generate_req.add_argument(
        '-o',
        '--output',
        default='requirements.txt',
        help='Save generated requirements to the specified file (requirements.txt by default)'
    )
    parser_generate_req.add_argument(
        'files',
        metavar='FILE',
        nargs='+',
        help='Root module for which the requirements will be generated'
    )
    parser_generate_req.set_defaults(func=generate_requirements)

    parser_job = subparsers_project.add_parser('job')
    subparsers_job = parser_job.add_subparsers(required=True)

    def add_project_id_argument(parser):
        parser.add_argument('-p', '--project-id', required=True, help='DataSphere project ID')

    def add_id_argument(parser, help: str = 'Job ID'):
        parser.add_argument('--id', required=True, help=help)

    def add_output_argument(parser, help: str = 'Output file (stdout by default)'):
        parser.add_argument('-o', '--output', help=help, default=sys.stdout, type=argparse.FileType('w'))

    def add_format_argument(parser):
        parser.add_argument(
            '--format', help='Output format',
            choices=[e.value for e in Format],
            default=Format.TABLE.value,
        )

    def add_async_argument(parser):
        parser.add_argument('--async', action='store_true', dest='async_', help='Async mode')

    async_execution_output_help = 'File with execution data, for async mode (none by default)'

    parser_execute = subparsers_job.add_parser('execute', help='Execute job')
    add_project_id_argument(parser_execute)
    parser_execute.add_argument('-c', '--config', required=True, help='Config file', type=argparse.FileType('r'))
    add_async_argument(parser_execute)
    add_output_argument(parser_execute, help=async_execution_output_help)
    parser_execute.set_defaults(func=execute)

    parser_fork = subparsers_job.add_parser('fork', help='Execute job using existing one as a template')
    add_id_argument(parser_fork, help='Source job ID')
    parser_fork.add_argument('--arg', action='append', help='New argument value', metavar='NAME=VALUE')
    parser_fork.add_argument('--var', action='append', help='New path for variable', metavar='NAME=PATH')
    parser_fork.add_argument('--name', help='New job name')
    parser_fork.add_argument('--desc', help='New job description')
    parser_fork.add_argument('--env-var', action='append', help='Environment variable', metavar='NAME=VALUE')
    parser_fork.add_argument('--docker-image-id', help='ID of Docker image resource')
    parser_fork.add_argument('--docker-image-url', help='URL of Docker image')
    parser_fork.add_argument('--docker-image-username', help='Username for Docker image')
    parser_fork.add_argument('--docker-image-password-secret-id', help='Secret name of password for Docker image')
    parser_fork.add_argument('--working-storage-type', help='Working storage disk type')
    parser_fork.add_argument('--working-storage-size', help='Working storage disk size')
    parser_fork.add_argument(
        '--cloud-instance-type',
        action='append',
        help='Computing resource configuration for executing the job. '
             'In case of multiple instance types are passed, they will be resolved in the specified order',
    )
    add_async_argument(parser_fork)
    add_output_argument(parser_fork, help=async_execution_output_help)
    parser_fork.set_defaults(func=fork)

    parser_attach = subparsers_job.add_parser('attach', help='Attach to the job execution')
    add_id_argument(parser_attach)
    parser_attach.set_defaults(func=attach)

    parser_list = subparsers_job.add_parser('list', help='List jobs')
    add_project_id_argument(parser_list)
    add_output_argument(parser_list)
    add_format_argument(parser_list)
    parser_list.set_defaults(func=ls)

    parser_get = subparsers_job.add_parser('get', help='Get job')
    add_id_argument(parser_get)
    add_output_argument(parser_get)
    add_format_argument(parser_get)
    parser_get.set_defaults(func=get)

    parser_delete = subparsers_job.add_parser('delete', help='Delete job')
    add_id_argument(parser_delete)
    # parser_delete.set_defaults(func=delete)  # DATASPHERE-1339

    parser_cancel = subparsers_job.add_parser('cancel', help='Cancel job')
    add_id_argument(parser_cancel)
    parser_cancel.add_argument("--graceful", "-g", action='store_true', help="Shutdown gracefully")
    parser_cancel.set_defaults(func=cancel)

    parser_project_get = subparsers_project.add_parser('get', help='Get project')
    add_id_argument(parser_project_get, help='Project ID')
    add_output_argument(parser_project_get)
    add_format_argument(parser_project_get)
    parser_project_get.set_defaults(func=get_project)

    parser_project_list = subparsers_project.add_parser('list', help='List projects')
    parser_project_list.add_argument('-c', '--community-id', required=True, help='Community ID')
    add_output_argument(parser_project_list)
    add_format_argument(parser_project_list)
    parser_project_list.set_defaults(func=ls_projects)

    parser_set_data_ttl = subparsers_job.add_parser('set-data-ttl', help='Set job data TTL')
    add_id_argument(parser_set_data_ttl)
    # maybe we need argument with some string timedelta format, i.e. "1y100d10h" (1 year, 100 days and 10 hours)
    parser_set_data_ttl.add_argument('--days', required=True, help='Data TTL days or "infinity"', type=str)
    parser_set_data_ttl.set_defaults(func=set_data_ttl)

    parser_download_files = subparsers_job.add_parser('download-files', help='Download job files')
    add_id_argument(parser_download_files)
    parser_download_files.add_argument('--with-logs', help='Download log files', action='store_true')
    parser_download_files.add_argument('--with-diagnostics', help='Download diagnostics files', action='store_true')
    parser_download_files.add_argument('--output-dir', help='Download all files to the specified directory', type=str)
    parser_download_files.set_defaults(func=download_job_files)

    return parser_datasphere


def main():
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()
    sdk = SDK(args.token, args.profile, args.log_level, args.log_config, args.log_dir, args.user_agent)

    try:
        args.func(args, sdk)
    except Exception as e:
        # TODO: move exception handling to SDK
        log_exception(e, sdk.logs_file_path)
        print_logs_files(sdk.logs_file_path)
        raise
    finally:
        check_package_version()


logger = logging.getLogger('datasphere.main')
# logger for things which do not go through logging, such as traceback in stderr
logger_file = logging.getLogger('datasphere_file')


def log_exception(e: Exception, logs_file_path: str):
    title = 'Error occurred'
    md = None
    if isinstance(e, grpc.RpcError):
        md = e.args[0].initial_metadata
        title = 'RPC error occurred'
    elif isinstance(e, OperationError):
        md = e.call_which_created_op.initial_metadata()
        title = 'Operation error occurred'
    elif isinstance(e, ProgramError):
        title = 'Program error occurred'
    md_str = '\n\theaders\n' + '\n'.join(f'\t\t{h.key}: {h.value}' for h in md) if md else ''
    logger.error('%s\n\tlogs file path: %s%s', title, logs_file_path, md_str)
    logger_file.exception(e)


if __name__ == '__main__':
    main()
