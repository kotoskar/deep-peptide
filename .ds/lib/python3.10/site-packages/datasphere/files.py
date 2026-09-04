import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Union, BinaryIO, Optional
from zipfile import ZipFile

import requests

from datasphere.api import jobs_pb2 as jobs
from datasphere.config import VariablePath, local_module_prefix
from datasphere.pyenv import PythonEnv
from datasphere.utils import humanize_bytes_size

logger = logging.getLogger(__name__)


def prepare_inputs(inputs: List[VariablePath], tmpdir: str) -> List[VariablePath]:
    prepared_inputs = []
    for input_ in inputs:
        input_path = Path(input_.path)
        if not input_path.is_dir():
            prepared_inputs.append(input_)
            continue
        logger.debug('zip local dir `%s` ...', input_)
        with tempfile.NamedTemporaryFile('rb', dir=tmpdir, delete=False) as tmp_file:
            zip_path(input_path, tmp_file)
            path = VariablePath(
                str(input_path),
                var=input_.var,
                compression_type=jobs.FileCompressionType.ZIP,
                _archive_path=tmp_file.name,
            )
            path.get_file(tmp_file)
            prepared_inputs.append(path)
    return prepared_inputs


def prepare_local_modules(py_env: PythonEnv, tmpdir: str) -> List[VariablePath]:
    result = []
    for i, module in enumerate(py_env.local_modules_paths):
        logger.debug('zip local module `%s` ...', module)
        with tempfile.NamedTemporaryFile('rb', dir=tmpdir, delete=False) as ar:
            zip_path(module, ar)

            # Path does not matter for local module since it will be unzipped to correct location, also, lzy
            # determines local module path as absolute path in general case, so we give it utility var value.
            path = VariablePath(
                ar.name,
                var=f'{local_module_prefix}_{i}',
                compression_type=jobs.FileCompressionType.ZIP,
            )
            path.get_file(ar)

            result.append(path)

    return result


def zip_path(path: Union[str, Path], zip_fileobj: BinaryIO) -> None:
    with ZipFile(zip_fileobj.name, 'w') as z:
        _zip_path(path, z)

    zip_fileobj.seek(0)


def _zip_path(path: Union[str, Path], zip_file: ZipFile) -> None:
    path = Path(path)
    relative_to = path.parent

    paths: List[Path] = []
    if path.is_dir():
        for root, _, files in os.walk(path):
            paths.extend(Path(root) / filename for filename in files)
    else:
        paths.append(path)

    for path_at_fs in paths:
        path_to_write = path_at_fs.relative_to(relative_to)
        zip_file.write(path_at_fs, path_to_write)


def _get_total_size(files: List[jobs.StorageFile]) -> str:
    return humanize_bytes_size(sum(f.file.size_bytes for f in files))


def upload_files(files: List[jobs.StorageFile], inputs: List[VariablePath], sha256_to_display_path: Dict[str, str]):
    # Maybe add debug log about already uploaded files.
    if len(files) == 0:
        logger.info('no files to upload')
        return
    logger.info('uploading %d files (%s) ...', len(files), _get_total_size(files))
    path_to_effective_paths = {input_.path: input_.effective_path for input_ in inputs}
    for f in files:
        path = path_to_effective_paths.get(f.file.desc.path, f.file.desc.path)
        with open(path, 'rb') as fd:
            display_path = sha256_to_display_path.get(f.file.sha256, f.file.desc.path)
            logger.debug('uploading file `%s` (%s) ...', display_path, humanize_bytes_size(f.file.size_bytes))
            if not f.url:
                continue
            resp = requests.put(f.url, data=fd)
            resp.raise_for_status()
    logger.info('files are uploaded')


DOWNLOAD_FILES_MAX_TOTAL_SIZE_BYTES = 1 * (1 << 30)  # 1Gb


def download_files(files: List[jobs.StorageFile], output_dir: Optional[str] = None, job_link: Optional[str] = None):
    files = _filter_files_by_max_total_size(files, job_link)
    if len(files) == 0:
        logger.info('no files to download')
        return
    logger.info('downloading %d files (%s) ...', len(files), _get_total_size(files))
    for f in files:
        _download_file(f, output_dir)
    logger.info('files are downloaded')


def _filter_files_by_max_total_size(
        files: List[jobs.StorageFile],
        job_link: Optional[str] = None
) -> List[jobs.StorageFile]:
    total_size = 0
    filtered_files: List[jobs.StorageFile] = []
    files = sorted(files, key=lambda file: file.file.size_bytes)
    for f in files:
        total_size += f.file.size_bytes
        if total_size > DOWNLOAD_FILES_MAX_TOTAL_SIZE_BYTES:
            logger.info(
                'file `%s` will not be downloaded due to exceeding the total size limit of downloading files %s',
                f.file.desc.path,
                humanize_bytes_size(DOWNLOAD_FILES_MAX_TOTAL_SIZE_BYTES)
            )
        else:
            filtered_files.append(f)

    if len(filtered_files) != len(files):
        if job_link is not None:
            logger.info('you can download the skipped files from the job page: %s', job_link)
        else:
            logger.info('you can download the skipped files from the job page')

    return filtered_files


def _download_file(file: jobs.StorageFile, output_dir: Optional[str] = None):
    logger.debug('downloading file `%s` (%s) ...', file.file.desc.path, humanize_bytes_size(file.file.size_bytes))
    try:
        resp = requests.get(file.url)
        resp.raise_for_status()
        path = Path(file.file.desc.path)
        if output_dir:
            path = Path(output_dir) / path.name
        if not path.parent.exists():
            # Create dirs containing output file.
            path.parent.mkdir(parents=True, exist_ok=True)
        download_path = path.with_suffix(path.suffix + '.download')
        try:
            with download_path.open('wb') as fd:
                for chunk in resp.iter_content(chunk_size=1 << 24):  # 16Mb chunk
                    fd.write(chunk)
            _process_downloaded_file(file.file, download_path)
        finally:
            shutil.rmtree(download_path, ignore_errors=True)
    except Exception as e:
        logger.warning(f'cannot download file {file.file.desc.path} ({file.file.desc.var})')
        logger.exception(e)


_UNCOMPRESSED_FILE_TYPES = (jobs.FileCompressionType.NONE, jobs.FileCompressionType.FILE_COMPRESSION_TYPE_UNSPECIFIED)


def _process_downloaded_file(file: jobs.File, path: Path) -> None:
    if file.compression_type in _UNCOMPRESSED_FILE_TYPES:
        path.replace(path.with_suffix(''))  # remove .download suffix
    elif file.compression_type is jobs.FileCompressionType.ZIP:
        zip_file = path.with_suffix('.zip')
        logger.debug("File `%s` is compressed with ZIP. Rename to `%s`", path, zip_file)
        path.rename(zip_file)
        logger.debug("Uncompressing ZIP file `%s`", zip_file)
        _unzip(zip_file)
        logger.debug("ZIP file `%s` uncompressed", zip_file)
        zip_file.unlink()
        logger.debug("Original ZIP file `%s` removed", zip_file)


def _unzip(zip_file: Path) -> None:
    with ZipFile(zip_file, 'r') as zf:
        zf.extractall(zip_file.parent)
