from datetime import datetime
import logging.config
import os
from pathlib import Path
from string import Template
import tempfile
from typing import Optional, TextIO
import yaml

logger_system_log = logging.getLogger('system.log')
logger_docker_stats = logging.getLogger('docker_stats.tsv')
logger_gpu_stats = logging.getLogger('gpu_stats.tsv')
logger_program_stdout = logging.getLogger('program_stdout')
logger_program_stderr = logging.getLogger('program_stderr')
logger_job_progress = logging.getLogger('job_progress.jsonl')


def configure_logging(level: str, user_config: Optional[TextIO], log_dir: Optional[str]) -> str:
    if log_dir:
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            log_dir_path.mkdir(parents=True, exist_ok=True)
        elif not log_dir_path.is_dir():
            raise ValueError(f'log directory is a file: {log_dir}')
        log_file_path = log_dir
    else:
        ts = datetime.now().isoformat()
        if os.name == 'nt':
            ts = ts.replace(':', '-')  # Windows do not allow colons in file names
        log_file_path = Path(tempfile.gettempdir()) / 'datasphere' / f'job_{ts}'
        log_file_path.mkdir(parents=True, exist_ok=True)
    if user_config:
        cfg_str_tpl = user_config.read()
    else:
        cfg_str_tpl = Path(__file__).with_name('logging.yaml').read_text()
    cfg_str = Template(cfg_str_tpl).substitute({'LOG_LEVEL': level, 'LOG_FILE_PATH': str(log_file_path)})
    cfg = yaml.safe_load(cfg_str)
    logging.config.dictConfig(cfg)
    return str(log_file_path)
