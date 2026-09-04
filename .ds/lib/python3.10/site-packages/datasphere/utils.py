import itertools
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import logging
import os

from envzy import AutoExplorer
from packaging.version import parse as version_parse
from pathlib import Path
import re
from shutil import which
import subprocess
from typing import BinaryIO, Optional, Tuple, List, Dict, Any

from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

from envzy.pypi import PYPI_INDEX_URL_DEFAULT, get_project_page, get_pypi_client

logger = logging.getLogger(__name__)


def get_sha256_and_size(f: BinaryIO) -> Tuple[str, int]:
    h = hashlib.sha256()
    sz = 0

    for chunk in iter(lambda: f.read(65_536), b''):
        h.update(chunk)
        sz += len(chunk)

    return h.hexdigest(), sz


def query_yes_no(question: str, default: Optional[bool] = True) -> bool:
    prompt = {True: 'Y/n', False: 'y/N', None: 'y/n'}[default]
    options = {'yes': True, 'y': True, 'no': False, 'n': False}
    while True:
        choice = input(f'{question} [{prompt}]: ').lower()
        if default is not None and choice == '':
            return default
        elif choice in options:
            return options[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def humanize_bytes_size(size: int) -> str:
    for unit in ('', 'K', 'M', 'G', 'T', 'P', 'E', 'Z'):
        if abs(size) < 1024.0:
            return f'{size:3.1f}{unit}B'
        size /= 1024.0
    return f'{size:.1f}YB'


def parse_human_size(size: str) -> int:
    size = size.strip().upper()
    if size.isdigit():
        return int(size)

    for unit, mult in [('KB', 1 << 10), ('MB', 1 << 20), ('GB', 1 << 30), ('TB', 1 << 40), ('PB', 1 << 50)]:
        if size.endswith(unit):
            size = size[:-len(unit)].strip()
            return int(size) * mult

    raise ValueError(f'Invalid size: {size}')


package = 'datasphere'
version_pattern = re.compile(r'\d+\.\d+\.\d+')


def check_package_version():
    current = importlib.metadata.version(package)

    # clear envzy cache
    get_pypi_client.cache_clear()
    get_project_page.cache_clear()

    project_page = get_project_page(pypi_index_url=PYPI_INDEX_URL_DEFAULT, name=package)

    # filter release candidates and other non-release staff
    release_packages = [p for p in project_page.packages if version_pattern.fullmatch(p.version)]

    if len(release_packages) == 0:
        logger.warning('No released packages found to check CLI version')
        return

    # latest package should be last in list, but let's sort by version lexicographically, just in case
    latest_package = sorted(release_packages, key=lambda p: version_parse(p.version))[-1]
    latest = latest_package.version

    if current != latest:
        logger.warning('Installed version of CLI is %s, and the latest version is %s, please update CLI using '
                       '`pip install -U %s`', current, latest, package)


@dataclass(frozen=True)
class PythonModule:
    path: str
    is_name: bool = False  # Module can be specified with `python -m <name> ...`


# Possible suffixes [dmu] came from https://peps.python.org/pep-3149/#proposal
common_python_name_re = re.compile(r'python([23](\.\d{1,2})?([dmu])?)?(\.exe)?')


def parse_python_main_module(cmd: str) -> Optional[PythonModule]:
    args = cmd.split()
    if len(args) == 0:
        raise ValueError('`cmd` is empty')
    first_arg = args[0]

    # First argument must be executable file: either binary file, python interpreter or script.

    # At first, let's check if python interpreter path is at first argument. If it is, then main module can be specified
    # as `python <main_module_path> ...` or `python -m <main_module_name> ...`
    python_is_first_arg = False
    if is_common_python_name(first_arg):
        logger.debug('`%s` is detected as Python interpreter path by common name', first_arg)
        python_is_first_arg = True
    elif is_python_interpreter(first_arg):
        # User can set some custom slug for python.
        python_is_first_arg = True

    if python_is_first_arg:
        if len(args) < 2:
            raise ValueError(f'`{first_arg}` must be followed by other arguments')
        if '-m' in args:
            # special case, main module name is in `-m`
            return PythonModule(path=args[args.index('-m') + 1], is_name=True)
        # common case, main module path is at first positional argument
        # TODO: skip python interpreter flags/options
        return PythonModule(path=args[1])

    # May be main script path is at first argument itself. It can be:
    # - Usual .py file with Python, or
    # - Auto-generated console script entry point (when you define `entry_points` in your setup.py).
    # In each case, there must be a Python shebang at the first line of file.
    # Read more about Python shebangs here – https://realpython.com/python-shebang/.

    # It could be executable name from $PATH, so first search for it.
    first_arg = which(first_arg) or first_arg

    with open(first_arg) as f:
        try:
            # Limit number of chars to read from first line, in case if some big binary file accidentally
            # appears in python `cmd`.
            first_line = f.readline(1024).strip()
        except UnicodeDecodeError:
            # It seems to be binary executable file
            logger.debug('`%s` is non-unicode file so it is not a Python program', cmd)
            return None
    if first_line.startswith('#!'):
        hint = first_line[2:].strip()
        hint_path = Path(hint)
        is_python_shebang = (
            # Commonly used python shebangs
            hint.startswith('/usr/bin/env python') or
            hint.startswith('/usr/bin/env -S python') or
            # Shebang in venv's entry points looks like `#!/pyenv/versions/3.8.17/envs/lzy/bin/python3.8`.
            # It could be detected by next check, but this one is quicker.
            hint_path.exists() and is_common_python_name(hint_path.parts[-1]) or
            # User can set some custom slug for python.
            is_python_interpreter(hint, raise_not_exist=False)
        )
        if is_python_shebang:
            logger.debug('shebang `%s` in file `%s` is detected as Python script', first_line, first_arg)
            return PythonModule(path=first_arg)

    # We can't recognize cmd as Python program.
    logger.debug('`%s` is not recognized as Python program', cmd)
    return None


def is_common_python_name(name: str) -> bool:
    return common_python_name_re.fullmatch(name.lower()) is not None


def is_python_interpreter(path: str, raise_not_exist: bool = True) -> bool:
    # Try to execute file with `--version` and check if stdout matches with python interpreter version info.
    try:
        stdout = subprocess.check_output([path, '--version'], timeout=1)
    except FileNotFoundError:
        if raise_not_exist:
            raise ValueError(f'file `{path}` was not found')
        else:
            return False
    except subprocess.CalledProcessError:
        return False  # Timeout or invalid command format
    except subprocess.TimeoutExpired:
        return False
    except OSError as e:
        if e.errno == 8:  # Exec format error – if file has mode +x but non-executable format
            return False
        else:
            raise
    if b'Python' in stdout:
        logger.debug('`%s` is detected as Python interpreter by `%s --version` stdout', path, path)
        return True
    return False


def print_logs_files(logs_dir: str):
    logger.info('see job log files at %s', logs_dir)
    for f in os.listdir(logs_dir):
        path = Path(logs_dir) / f
        dir_suffix = '/' if path.is_dir() else ''
        file_size = f'  ({humanize_bytes_size(os.path.getsize(path))})' if path.is_file() else ''
        logger.info('  * %s%s%s', f, dir_suffix, file_size)


def timedelta_to_pb_duration(td: timedelta) -> Duration:
    seconds = int(td.total_seconds())
    data_ttl = Duration()
    data_ttl.FromSeconds(seconds)
    return data_ttl


def datetime_to_pb_timestamp(dt: datetime) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


DURATION_REGEX = re.compile(r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?(?:(\d+)ms)?')


def parse_human_duration(time_str: str) -> Optional[timedelta]:
    match = DURATION_REGEX.fullmatch(time_str)
    if not match:
        return None

    days, hours, minutes, seconds, millis = match.groups()

    days = int(days) if days is not None else 0
    hours = int(hours) if hours is not None else 0
    minutes = int(minutes) if minutes is not None else 0
    seconds = int(seconds) if seconds is not None else 0
    millis = int(millis) if millis is not None else 0

    duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=millis)
    return duration


def get_requirements_for_modules(modules: List[PythonModule], explorer: AutoExplorer) -> List[str]:
    requirements = []
    for module in modules:
        sys.path.append(str(Path(module.path).parent))
        namespace = _get_module_namespace(module)
        requirements.append([
            (f'{name}=={version}' if version else name) for name, version in
            explorer.get_pypi_packages(namespace).items()
        ])

    return merge(requirements)


def get_local_modules_paths(modules: List[PythonModule], explorer: AutoExplorer) -> List[str]:
    local_modules_paths = []
    for module in modules:
        sys.path.append(str(Path(module.path).parent))
        namespace = _get_module_namespace(module)
        local_modules_paths.append(explorer.get_local_module_paths(namespace))

    return merge(local_modules_paths)


main_module_checks = [
    "if __name__ == '__main__':",
    'if __name__ == "__main__":',
]


def _get_module_namespace(module_params: PythonModule) -> Dict[str, Any]:
    if module_params.is_name:
        module = importlib.import_module(module_params.path)
    else:
        # User scripts shouldn't execute any code in global namespace (only imports and declarations) and should contain
        # `if __name__ == '__main__'` check in main module, otherwise on module import user program will start to
        # execute, but we only want to collect global namespace.
        # If user does it anyway, there is no straightforward and cross-platform way to import module with timeout, in
        # case if some long-running calculations will start.
        # CLI commands, generated by setup.py `entry_points`, also have main module check.
        # TODO: root module launched with multiprocessing will have different name, not '__main__'
        #   this is an exotic use-case, fix this if such use-case will appear on practice
        code_lines = Path(module_params.path).read_text().split('\n')
        has_main_module_check = any(
            any(line.startswith(check) for check in main_module_checks)
            for line in code_lines
        )
        if not has_main_module_check:
            raise ValueError(f'Main script `{module_params.path}` must have line `{main_module_checks[0]}`')
        module_spec = importlib.util.spec_from_file_location('module', module_params.path)
        if module_spec is None:
            raise ValueError(f'Main script `{module_params.path}` can\'t be loaded by importlib')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    return vars(module)


def merge(items: List[List[str]]) -> List[str]:
    return list(sorted(set(  # drop duplicates and sort
        itertools.chain(*items)  # flatten
    )))
