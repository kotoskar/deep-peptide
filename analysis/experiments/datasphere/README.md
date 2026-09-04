# fp32 baseline re-run on Yandex DataSphere

`runs/5cv_baseline_esm2` is the only run in the nested-CV grid trained with bf16
autocast; the four variants are fp32. Every effect the paper reports is measured
against that baseline, so every effect is confounded with a change of training
precision. This job re-runs the baseline with `amp` pinned to `false`.

20 cells, ~6 GPU-hours each. At `CONC=8` on one A100 that is ~15 h of wall-clock
and about **8,100 RUB** on `g2.1` (542.88 RUB/h, VAT included). `g2.2` (2x A100,
`CONC=16 GPUS=0,1`) costs about the same in total and finishes in half the time.

## Check these three things BEFORE uploading anything

1. **GPU quota.** `g2.1`/`g2.2`/`g2.4` ship with a VM-usage quota of **0**. The
   docs footnote says to raise it you must top the billing account up by at
   least $10 or open a support request, and the turnaround is not published.
   Check https://console.yandex.cloud/cloud?section=quotas and confirm a
   non-zero value before anything else. This is the single most likely blocker,
   and a grant-funded balance may not count as a top-up.
2. **The configuration must be allowed in the community.** Community ->
   Restrictions -> Configurations gates which instance types a project may use.
   Needs the `datasphere.communities.admin` role.
3. **Billing account attached to the community** (not the project), state
   `ACTIVE` or `TRIAL_ACTIVE`.

Also: at most 10 active jobs, and the CLI will not download more than 1 GiB of
outputs — which is why the driver returns a ~100 MB `results.tar` rather than
the whole run directory.

## Prepare the inputs

The 8,897 embedding files the 2026 ESM-2 split needs come to 9.19 GiB. They ship
as three tars because a single job input file is capped at 5 GiB and the client
zips inputs **without compression**, so one 9.2 GiB entry is rejected outright.

```bash
env/bin/python analysis/experiments/datasphere/pack_embeddings.py   # writes /home/oskar/ds_job/emb_part{0,1,2}.tar
cp /home/oskar/ds_job/emb_part*.tar .                               # next to the repo root, where config.yaml expects them
```

Total input is 9.19 GiB against a documented per-job data cap of 10 GB. The docs
do not say whether that cap counts inputs alone or inputs plus outputs, logs and
cache; if the job is rejected at submit time, switch to the S3 route below. A
rejection is immediate and costs nothing, which is why this route is worth
trying first.

## Run

```bash
python3 -m venv .ds && . .ds/bin/activate && pip install datasphere
yc init                     # opens a browser; or export YC_TOKEN=<oauth token>

# 1. smoke test first: one epoch, two cells. Catches a wrong CUDA build, a
#    missing package or a bad path for a few tens of roubles instead of 8,000.
datasphere project job execute -p <PROJECT_ID> -c analysis/experiments/datasphere/config_smoke.yaml

# 2. the real thing
datasphere project job execute -p <PROJECT_ID> -c analysis/experiments/datasphere/config.yaml
```

The job keeps running server-side if the local CLI session drops; reattach with
`datasphere project job list -p <PROJECT_ID>` and `... job logs --id <JOB_ID>`.

## If the input route is rejected

Put the same three tars in Object Storage and mount them instead:

1. Create a bucket, a service account with `storage.editor`, and a **static
   access key** for it.
2. `aws --endpoint-url=https://storage.yandexcloud.net s3 cp emb_part0.tar s3://<bucket>/emb/` (and 1, 2).
3. Project -> Create resource -> **S3 connector**: endpoint
   `https://storage.yandexcloud.net/`, the bucket, a mount name, the static key,
   mode **Read and write**.
4. In `config.yaml` drop the three `emb_part*.tar` lines from `inputs`, add
   `s3-mounts: [<connector-id>]`, and change the extract loop in `run_job.sh` to
   read from `/job/s3/<connector-id>/emb/`.

Do **not** point the training run at the S3 mount directly: the docs warn that a
flat directory with many files over FUSE degrades badly, and this run reads 8,897
of them from 8 concurrent workers. Extract to local disk first, as the driver does.

## What comes back

`results.tar` holds, per cell: `cell_result.json`, `config.json`, the metrics
JSONs and `model.pt`. The periodic checkpoints stay behind; they exist only to
resume a dying cell. Rescoring with the corrected matcher happens at home from
`model.pt`.

## Unverified

Nothing here has been run end to end. Specifically unconfirmed: whether a
DataSphere job has a maximum wall-clock duration (nothing is documented, and 15 h
is long), the CUDA driver version on the job VM against `torch 2.8.0+cu128`, and
whether the 10 GB job-data cap counts outputs and cache alongside inputs. The
smoke config exists to settle the first two cheaply.
