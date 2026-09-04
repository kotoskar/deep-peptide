import itertools
from dataclasses import dataclass
import importlib
import logging
import os
from packaging.requirements import Requirement
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List
import yaml

from envzy import AutoExplorer, ModulePathsList

from datasphere.config import PythonEnv as PythonEnvConfig
from datasphere.utils import PythonModule, get_local_modules_paths, get_requirements_for_modules

logger = logging.getLogger(__name__)


@dataclass
class PythonEnv:
    version: str
    local_modules_paths: ModulePathsList
    requirements: List[str]
    pip_options: Optional[PythonEnvConfig.PipOptions] = None

    def to_string(self) -> str:
        result_str = f"python_version: {self.version}\n"
        if self.pip_options:
            # Requirements from envzy explorer only contains precise packages versions.
            # User-defined requirements can contain anything supported by pip, so clash with `pip-options`
            # is possible (not solved by now).
            result_str += 'pip:\n'
            result_str += f'  index-url: {self.pip_options.index_url}\n'
            result_str += f'  extra-index-urls: {self.pip_options.extra_index_urls}\n'
            result_str += f'  trusted-hosts: {self.pip_options.trusted_hosts}\n'
            result_str += f'  no-deps: {self.pip_options.no_deps}\n'
        if self.requirements:
            result_str += f'requirements: {list(self.requirements)}'
        return result_str


def define_py_env(root_modules: List[PythonModule], py_env_cfg: PythonEnvConfig) -> PythonEnv:
    version = None
    local_modules_paths = None
    requirements = None

    if not py_env_cfg.is_fully_manual:
        # User may not add cwd to PYTHONPATH, in case of running execution through `datasphere`, not `python -m`.
        # Since path to python script can be only relative, this should always work.
        sys.path.append(os.getcwd())

        extra_index_urls = []
        if py_env_cfg.pip_options and py_env_cfg.pip_options.extra_index_urls:
            extra_index_urls = py_env_cfg.pip_options.extra_index_urls

        explorer = AutoExplorer(extra_index_urls=extra_index_urls)
        version = '.'.join(str(x) for x in explorer.target_python)

        # We can't launch explorer once on namespace merged from multiple modules, because vars from different modules
        # can override each other. So instead we merge explorer artifacts – local modules and requirements.
        local_modules_paths = get_local_modules_paths(root_modules, explorer)
        requirements = get_requirements_for_modules(root_modules, explorer)
        logger.debug('auto-defined python env:\n\tlocal modules: %s\n\trequirements: %s',
                     local_modules_paths, requirements)

    return PythonEnv(
        version=py_env_cfg.version if py_env_cfg.version else version,
        requirements=_parse_requirements(py_env_cfg.requirements) if py_env_cfg.requirements else requirements,
        local_modules_paths=py_env_cfg.local_modules_paths if py_env_cfg.local_modules_paths else local_modules_paths,
        pip_options=py_env_cfg.pip_options,
    )


# Allow packages specifiers (with extras) and flags/options (supported by server).
def _parse_requirements(f: Path) -> List[str]:
    lines = [line.strip() for line in f.read_text().strip().split('\n') if line.strip()]
    for line in lines:
        if line == '--no-deps':
            continue
        if line.startswith('--extra-index-url'):
            continue
        if line.startswith('--trusted-host'):
            continue
        req = Requirement(line)
        if req.marker:
            raise ValueError(f'requirement markers are not supported ({line})')
        if req.url:
            raise ValueError(f'requirement url is not supported ({line})')
    return lines
