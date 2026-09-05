# fp32 baseline re-run — state as of 2026-09-05 01:30 (UTC+3)

## Running now

Two jobs, both on their own single A100.

**Job `bt1j9t5c56k5edf1grcs`**, `config.yaml` unchanged from what is committed:
20 cells of the 5×4 nested CV, ESM-2 base architecture, `amp=false`, one A100,
`MPS=1 THREADS=2 CONC=10`, two waves of ten.

* expected ~18 h of A100 time, so finishing around **19:00-20:00 on 5 Sept**
* ~9.8k ₽ at 542.88 ₽/h
* output lands in the repo root as `results.tar`, containing
  `runs/5cv_baseline_esm2_fp32/outer*_inner*/` (cell_result, config, metrics, model.pt)
  and `logs/` including `usage.tsv`

Why it exists: `runs/5cv_baseline_esm2` is the only run in the grid trained with bf16
autocast, and every effect the paper reports is measured against it, so every effect is
confounded with a change of training precision. This is the clean comparison.

**Deadline rule.** MPS could not be confirmed from the live log -- `attach` returned nothing
for a running job, which is consistent with both a cold environment build and a silent MPS
failure, so it is not evidence either way. It does not need to be observed: with MPS the job
ends around 19:00-20:00 on 5 Sept, without it around 08:00 on 7 Sept, which is past the
submission. **So if the job is still EXECUTING at 22:30 on 5 Sept, plan as though MPS
failed** and take the text-only route for the bf16 confound (item 1 below), so the paper is
safe regardless. Whether to also cancel the job then is a spending decision: cancelling
saves the remaining hours but discards everything computed, since results upload only at the
end. There is no documented per-job wall-clock cap in the DataSphere limits, so the job is
not expected to be killed on its own.

One thing *is* observable without waiting: the console's resource graphs. **CPU load around
10-11 of 28 means MPS and the thread cap are both working** (that is what the winning
calibration showed); around 15 means MPS never started and the run is on the 55-hour path;
around 67 means MPS started but the thread cap did not. VRAM sits near 55 GB either way and
GPU utilisation reads ~100% in all three, so neither of those distinguishes anything.

### Job `bt12l0qo04avm3k1qdil` — the 2022-release baseline, insurance

Launched 09:02 MSK on 5 Sept from `config_2022.yaml`: the same 20 cells on the 2022 release,
`amp=false`, `MPS=1 THREADS=2 CONC=10`. ~15 h and ~8.3k RUB, so it should land around
midnight. It exists because the same run is being trained elsewhere and may not arrive; if
both arrive, they cross-check each other -- but note the handoff's own
`runs/baseline_esm2/config.json` carries `amp: true`, so the outside copy is bf16 unless it
was run with `EXTRA_SET="amp=false"`, while this one is fp32 by construction.

The driver is no longer tied to one release: `EMB_DIR`, `BASE_CFG`, `SPLIT_FILE`, `DATA_FILE`
and `EMB_COUNT` select it, and the embedding tars are found by glob rather than by a fixed
list of three. `pack_embeddings.py --release 2022` produced `emb22_part{0,1,2}.tar` (7167
unique sequences, 7.5 GiB).

## When it lands

The job was submitted with `--async`, so nothing arrives by itself; the archive has to be
pulled. `attach` replays the whole log and downloads the declared output into the repo root:

```bash
set -a; . ./.env; set +a; export YC_TOKEN="$DS_TOKEN"
datasphere project job attach --id bt1j9t5c56k5edf1grcs   # 2026 fp32 baseline
tar -xf results.tar                                       # into the repo root
datasphere project job attach --id bt12l0qo04avm3k1qdil   # 2022 baseline; overwrites results.tar
tar -xf results.tar
bash analysis/experiments/datasphere/ingest_any.sh        # auto-detects the new run
```

`ingest_any.sh` runs the corrected-matcher rescoring, then the tolerance sweep (needs the
local GPU, ~54 s per cell), then rebuilds the headline tables. A partial run is now handled:
`rescore_nested_cv_corrected.py --partial` aggregates over the outer folds whose four inner
cells are all present and marks the summary incomplete.

## What the calibration established

Three jobs, ten cells × three epochs each, same data, same instance:

| configuration | batch wall | per epoch | load (28 vCPU) |
|---|---|---|---|
| `CONC=10` | 3269 s | 962 s | 15.1 |
| `CONC=10 MPS=1` | 1668 s | 487 s | 67.7 |
| `CONC=10 MPS=1 THREADS=2` | 974 s | 264 s | 10.7 |

Raising concurrency alone does nothing: without MPS the driver time-slices between
processes, so ten cells deliver roughly one cell's throughput while 25 GB of VRAM and half
the cores sit idle. `utilization.gpu` reads 99-100% in every configuration including the
slow ones and is useless here. With MPS the cells finally run at once and immediately
oversubscribe the CPU, which the thread cap fixes. Together: 55 A100-hours → 18.

`g2.2` (two cards, same rubles, half the wall clock) is refused outright —
`Instance types g2.2 are not available for your community`.

## Still open, deliberately untouched

The paper text is frozen until the experiments land (your instruction). Waiting on a
decision:

1. **The bf16 disclosure.** If this run lands clean, the confound disappears from the main
   table and the text needs no caveat at all. If it does not land, the free empirical bound
   still applies: `train_run_esm2` (bf16) scored 0.6073 against `train_run_esm2_100` (fp32)
   0.5881 on the same 2022 split and hyperparameters, so bf16 scored *higher* and the
   reported effects are conservative. One seed each, and 0.019 is the size of the effects
   themselves, so it is a bound and not a measurement.
2. **The interval-overlap double standard** at `main.tex:305`.
3. **The ESM-C protein-set disclosure** — that row is on 8,999 proteins, the ESM-2 rows on
   8,897.

Also unapplied: 68 unadjudicated review candidates in `texs/ai4dd/PENDING_REVIEW.md`
(consistency 19, claims 16, bibliography 7, proofreading 26).

## Expected from elsewhere

Two deliveries were due overnight from outside this machine and arrive through chat, not on
disk: 60 ESM-C 6B cells (three rungs) and 20 cells of the 2022-release baseline. Unpack them
into `runs/` and the same `ingest_any.sh` call picks them up in one pass.

Warning carried over: `baseline2022_cv_handoff.tar` ships `runs/baseline_esm2/config.json`
with `amp: true`, so that delivery returns bf16 unless it was run with `EXTRA_SET="amp=false"`.
Worth confirming with whoever ran it before its numbers go in the table.
