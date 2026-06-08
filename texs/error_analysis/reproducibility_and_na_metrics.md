# Why some MCC/AUC cells were N/A — root cause and fix

The experiment tables (`analysis/canonical_metrics.md`) take P/R/F1 from each run's
train-time `test_metrics.json` but had to compute **MCC/AUC by a fresh inference
pass** (training never saved them). A fresh pass was trusted only if it reproduced
the run's train-time P/R/F1 within 0.015 — otherwise MCC/AUC were marked **N/A**, to
avoid pairing metrics from a different evaluation with the published P/R/F1.

Originally **~11 of 39 runs** were N/A. We root-caused and fixed it; the N/A set is
now just the 2 genuinely unrecoverable runs.

## Root cause: inference ran in bf16 AMP, training-time metrics were fp32

The divergence was **not** embeddings, non-determinism, or the ESM2 provider (all
earlier hypotheses, all wrong). It was a precision mismatch:

- Training's final test evaluation (`train_loop_crf.train()`,
  `run_dataloader(test_loader, …, desc='Test')`) **omits `use_amp`** → it runs in
  **fp32**. So the published P/R/F1 are fp32 numbers.
- `infer.py::evaluate_loader` read `amp=True` from each config and evaluated in
  **bf16 AMP**. bf16's coarser rounding shifts the model's logits slightly; for a
  **low-confidence model** that has many peptide boundaries sitting right at the ±3
  decision margin (the AFT runs, F1≈0.38–0.43), this flips a lot of borderline
  predictions — **precision swings ±0.05–0.13**, in both directions. For a confident
  model (esm2) almost nothing flips (≈0.001), which is why the baseline looked fine
  and hid the bug.

**Proof.** For `train_run_aft`, train-time saved F1=0.3818 / P=0.5294.
fp32 inference reproduces it **exactly** (0.3818 / 0.5294); bf16 inference gives
0.3703 / 0.4767 — exactly the old (N/A-triggering) infer value.

## Fix

`infer.py` now forces `use_amp=False` in evaluation (commit *fix infer.py: evaluate
in fp32*), matching how the training-time test metrics were produced. Re-inferring
all runs in fp32, the drift vs train-time **collapses to ~0.0000** for every
previously-diverging run (AFT, AFT+ESM2, 3Di, ESM-C, uniprot-2026 baseline). No
retraining, no embedding regeneration — the embeddings were always the same.

| run | drift before (bf16) | drift after (fp32) |
|---|---:|---:|
| train_run_aft | 0.061 | 0.000 |
| train_run_aft_no_lddt | 0.139 | 0.000 |
| train_run_aft_single | 0.113 | 0.000 |
| train_run_aft_plddt70 | 0.034 | 0.000 |
| train_run_esm2_aft | 0.020 | 0.000 |
| train_run_esm2_aft_no_lddt_gated | 0.049 | 0.000 |
| train_run_esm2+3di_proj | 0.037 | 0.000 |
| train_run_esmc_600m | 0.015 | 0.004 |
| uni2026_run_esm2 | 0.063 | 0.000 |

## Remaining N/A (2 runs) — genuinely need code, not a provider/precision fix

| run | reason | to recover |
|---|---|---|
| `esm2_aho_transition_bias_sparse_trainable_zero` | model class `lstmcnncrf_aho_transition_bias_sparse` was never committed to git | rewrite/restore the class, or retrain |
| `esm2_bond_loss_soft_l005_w5_tau15` | trained an old bond-only head (`bond_head.0.*`); the current `lstmcnncrf_boundary_bond_loss` class differs | restore the old class, or retrain |

(`train_run_esm2_25` is also non-trusted, but it is the mislabeled scaling run that
trained on the 50% file and is superseded by the corrected scaling series — it does
not need fixing.)

## On the "single ESM provider" question

`requirements.txt` does ship two ESM2 providers (`fair-esm` for the online path,
HuggingFace `transformers` for precomputed `embeddings_esm2`). This is real code
hygiene to tidy up, and it is the only thing that would matter for the **online/LoRA**
runs. But it turned out **not** to be the cause of any N/A cell — the precision bug
above explains all of them. So no provider-driven retraining is needed for the
results in the tables.
