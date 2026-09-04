import logging
from pathlib import Path
from typing import List, Set, Optional

from datasphere.config import Config
from datasphere.pyenv import PythonEnv

logger = logging.getLogger(__name__)


def _check_paths_do_not_contain_parents(paths: List[Path]) -> None:
    if len(paths) < 2:
        return
    unique_paths: Set[Path] = set(paths)
    if len(paths) != len(unique_paths):
        raise ValueError('Paths must be unique')
    for path in unique_paths:
        for parent in path.parents:
            if parent in unique_paths:
                raise ValueError(f"Path '{path}' is included in path '{parent}'")


def _check_local_modules_do_not_intersect_with_inputs(modules: List[Path], inputs: List[Path]) -> None:
    for input_ in inputs:
        input_parts = input_.parts
        for module in modules:
            module_parts = module.parts
            is_in_module = (
                len(input_parts) >= len(module_parts)
                and all(module_part == input_part for module_part, input_part in zip(module_parts, input_parts))
            )
            is_module_in = (
                len(module_parts) >= len(input_parts)
                and all(input_part == module_part for input_part, module_part in zip(input_parts, module_parts))
            )
            if is_in_module and is_module_in:
                logger.warning(f"Local module '{module}' duplicating input '{input_}'")
            elif is_in_module:
                logger.warning(f"Local module '{module}' includes input '{input_}'")
            elif is_module_in:
                logger.warning(f"Input'{input_}' includes module '{module}'")


def validate_inputs(config: Config, py_env: Optional[PythonEnv] = None) -> None:
    inputs = [Path(input_.path).absolute() for input_ in config.inputs]
    local_modules = [Path(path).absolute() for path in py_env.local_modules_paths] if py_env is not None else []
    _check_paths_do_not_contain_parents(inputs)
    _check_local_modules_do_not_intersect_with_inputs(local_modules, inputs)


def _is_relative_path(path: Path) -> bool:
    return not path.is_absolute() and '..' not in path.as_posix()


def validate_outputs(config: Config) -> None:
    """
    Since we don't know exact state of remote filesystem we can't check all output paths as absolute.
    So we can check path relative to PWD and absolute paths but not relative paths which escape PWD.
    """
    outputs = [Path(output.path) for output in config.outputs]
    relative_outputs = [output for output in outputs if _is_relative_path(output)]
    _check_paths_do_not_contain_parents(relative_outputs)
    absolute_outputs = [output for output in outputs if output.is_absolute()]
    _check_paths_do_not_contain_parents(absolute_outputs)


def validate_paths(config: Config, py_env: Optional[PythonEnv] = None) -> None:
    validate_inputs(config, py_env)
    validate_outputs(config)
