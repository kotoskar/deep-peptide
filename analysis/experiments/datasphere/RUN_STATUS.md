# fp32 baseline re-run — state as of 2026-09-05 01:30 (UTC+3)

## Running now

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

## When it lands

```bash
tar -xf results.tar                                     # into the repo root
bash analysis/experiments/datasphere/ingest_any.sh      # auto-detects the new run
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
