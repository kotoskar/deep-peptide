# Why some MCC/AUC cells are N/A, and what retraining would fix them

The experiment tables (`analysis/canonical_metrics.md`) leave MCC/AUC as **N/A for
13 of 39 runs**. A cell is N/A when a fresh inference pass (needed because the
training code never saved MCC/AUC) did **not** reproduce that run's train-time
P/R/F1 within 0.015 — so its freshly-computed MCC/AUC cannot be trusted to belong to
the same evaluation. This note explains the causes and exactly which runs are
affected, since the obvious hypothesis ("we trained on two different ESM2 providers")
turns out to explain only a small part.

## Root causes of train-time ↔ fresh-inference divergence

Investigated directly (see memory note `deeppeptide-infer-divergence`):

1. **Non-determinism (the dominant cause).** Two *identical* fresh inference passes
   of the same run gave different results — even ground-truth residue counts and loss
   drifted — because nothing seeded the RNGs or forced deterministic algorithms. The
   per-peptide ±3 metric is sensitive (it truncates labels to the decoded path
   length), so small perturbations flip matches. **Now fixed** (`src/utils/seeding.py`,
   wired into training + inference, `--seed`); fresh inference is bit-identical.
   But the *train-time* numbers in the tables were produced by the old
   non-deterministic code, so they still differ from a fresh pass.
2. **Embedding *content* differs between training and now (structural runs).** This
   is NOT about coverage — the AFT/3Di embeddings still cover ~96% of the test set
   (1482/1538 vs 1503 for esm2), stable. The issue is *values*: the AFT runs' configs
   pointed at `data/embeddings_aft*`, a path **deleted** in the reorg and remapped to
   `data/uniprot_2022/embeddings/embeddings_aft*`, whose contents are a **different
   generation** than the model trained on. Fed slightly different embedding vectors,
   the model produces a different set of predictions: recall stays ~stable (it still
   recovers the same true peptides) but **precision swings ±0.05–0.13, in both
   directions** (e.g. `train_run_aft` −0.053, `train_run_aft_no_lddt` +0.125) — the
   signature of a changed input, not noise. esm2's embeddings were never moved (same
   path) so it reproduces train-time to within 0.0005. (Inferred, not proven: the
   original AFT embeddings are gone, but the esm2-vs-AFT contrast and the
   precision-only, bidirectional swing point squarely at an input mismatch.)
3. **Two ESM2 providers coexist.** `requirements.txt` ships both `fair-esm` (used by
   the *online* ESM2 path, `embedding=online_esm2`) and HuggingFace `transformers`
   (`facebook/esm2_t33_650M_UR50D`, used by `make_embeddings.py` to build the
   *precomputed* `embeddings_esm2`). These give numerically different embeddings.
   This only matters where a run's inference embeddings differ from its training
   embeddings — in practice the **online/LoRA** runs.

## The 13 N/A runs, by cause

| # | category | what actually fixes it |
|--:|---|---|
| 7 | **Structural embeddings (AFT/3Di)** — fed a different embedding *generation* than trained on (path deleted+remapped in reorg); precision swings, recall stable | re-infer on the exact embeddings the run trained on, or retrain on the current ones, then re-infer. NOT an ESM2-provider issue; NOT a coverage issue (96% stable). |
| 2 | **Unrecoverable** — model class no longer in code | restore/rewrite the model class (or retrain), then infer. |
| 1 | **ESM-C** — different embedding model | re-infer deterministically; drift is borderline (0.0151). |
| 3 | **ESM2-based** | see below — only 1 is genuinely a provider issue. |

Runs in each category:

- **Structural (7):** `train_run_aft`, `train_run_aft_no_lddt`, `train_run_aft_single`,
  `train_run_aft_plddt70`, `train_run_esm2_aft`, `train_run_esm2_aft_no_lddt_gated`,
  `train_run_esm2+3di_proj`
- **Unrecoverable (2):** `esm2_aho_transition_bias_sparse_trainable_zero`,
  `esm2_bond_loss_soft_l005_w5_tau15`
- **ESM-C (1):** `train_run_esmc_600m`
- **ESM2-based (3):** `esm2_lora_lstmcnncrf`, `uni2026_run_esm2`, `train_run_esm2_25`

### The "ESM2 provider" question, precisely

Of the three ESM2-based N/A runs, the provider only cleanly applies to **one**:

- **`esm2_lora_lstmcnncrf`** (drift 0.023) — `embedding=online_esm2`, i.e. ESM2 is
  run **live via `fair-esm`**. This is the genuine single-provider case: pin one ESM2
  provider/version and retrain (or at least re-infer with the same provider used at
  training). The sibling `esm2_lora_lstmcnncrf_r4_last2_qv` (same online path) would
  be in the same boat.
- **`train_run_esm2_25`** (drift 0.042) — uses the **same** precomputed
  `embeddings_esm2` as the baseline, which diverges only 0.0005. Identical embeddings
  ⇒ **not** a provider problem; it is amplified non-determinism on a low-quality run.
  Moreover it is **superseded** by the corrected data-scaling series
  (`runs/scaling_trainfrac*`), so it does not need retraining at all.
- **`uni2026_run_esm2`** (drift 0.063) — a separate uniprot-2026 exploratory run with
  its own `emb_esm2`; lower priority and tied to the 2026 dataset, not the main 2022
  results.

## Bottom line / recommendation

- The headline hypothesis "many configs need retraining on a single ESM provider" is
  **mostly not the case**: the provider duality cleanly affects only the **online/LoRA**
  runs (1–2 configs). The other 11 N/A cells are non-determinism (now fixed in code)
  and structural-embedding coverage.
- To fill **all** N/A honestly, the clean path is to **retrain the affected runs with
  the new seeded/deterministic code on pinned embeddings** (one ESM2 provider for the
  online runs; regenerated full-coverage AFT/3Di embeddings for the structural ones),
  then a single deterministic inference supplies P/R/F1 **and** MCC/AUC from one pass.
- Cost-ranked retrain targets: (1) the 2 LoRA online-ESM2 runs — the only true
  provider case; (2) the 7 structural runs — but these need embedding regeneration
  first (AFT embeddings are ~200 h to recompute, so likely not worth it); (3) the 2
  unrecoverable runs need code, not just a provider. `esm2_25` needs nothing (dropped
  in favor of the corrected scaling series).
