# 0.10.0 (2025-01-17)

- New features:
  - Added support of DataSphere Spark Connector for DataSphere Jobs.
- Minor bugfixes and improvements.

# 0.9.0 (2024-12-18)

- New features:
  - Added pip options for manual python environment.
  - Added index-url for pip options
- Minor bugfixes and improvements.

# 0.8.0 (2024-11-20)

- New features:
  - Added command to generate requirements file – `datasphere generate-requirements`.
  - Added limit for downloading files size, see https://yandex.cloud/ru/docs/datasphere/concepts/limits
- Minor bugfixes and improvements.

# 0.7.9 (2024-09-27)

- Added possibility to generate output datasets, see `output-datasets` section in job config file.

# 0.7.6 (2024-08-19)

- Added possibility of job's graceful shutdown with `graceful-shutdown` section in job config file. It is also available
  for `cancel` command with option `-g/--graceful`.

# 0.7.4 (2024-06-24)

- Added command `download-files` for downloading completed job output files.

# 0.7.0 (2024-05-27)

- New features:
  -  Support for directories as input and output files.
  - `args` – arguments for job command line (`cmd`).
  - `fork` command, which allows to use another job as a template, overriding input/output files, 
     command line arguments, environment variables, docker image, working storage or cloud instance type.
  - Support of multiple cloud instance types (main and backoff-s).
  - Added GPU stats logger in `gpu_stats.tsv`.
- Minor bugfixes and improvements. 

# 0.6.12 (2024-04-03)

- Added commands to get and list projects – `project get`, `project list`.
- Added options for command output – `-o` to specify output file path (stdout by default) and `--format` to specify
  data format (tabular or json).

# 0.6.11 (2024-03-25)

- Support use case when main script is not in current working directory.
