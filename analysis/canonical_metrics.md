# Canonical Experiment Tables

> **Methodology:** P/R/F1 values are authoritative train-time values from `test_metrics.json` (or `homo_test_metrics.json` for Table 3). MCC and AUC are from fresh inference (`test_metrics_infer.json` / `homo_test_metrics_infer.json`), accepted only when `drift = max|train-time P/R/F1 − fresh P/R/F1|` ≤ 0.015. Hard overrides (always N/A): `esm2_bond_loss_soft_l005_w5_tau15` and `esm2_aho_transition_bias_sparse_trainable_zero` (model unrecoverable for infer). Values rounded to 3 decimal places. **Bold** = best in column (N/A cells excluded).

## Table 1: Architectural Changes (TEST set)

| Model | All |  |  |  |  | Peptides |  |  |  |  | Propeptides |  |  |  |  |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ESM2 (baseline) | 0.607 | 0.640 | **0.578** | **0.752** | 0.717 | 0.604 | 0.649 | **0.565** | **0.696** | 0.860 | 0.610 | 0.633 | **0.588** | **0.746** | 0.509 |
| ESM2 + telescopic CRF | 0.596 | 0.632 | 0.564 | 0.731 | 0.719 | **0.614** | 0.675 | 0.562 | 0.679 | **0.904** | 0.582 | 0.600 | 0.566 | 0.706 | 0.610 |
| ESM2 + Aho emission fusion | **0.615** | **0.686** | 0.558 | 0.737 | 0.646 | 0.594 | 0.676 | 0.529 | 0.682 | 0.822 | **0.633** | **0.693** | 0.582 | 0.735 | 0.411 |
| ESM2 + (Aho -> hidden layer 32) emission fusion | 0.596 | 0.674 | 0.534 | 0.733 | 0.779 | 0.595 | 0.657 | 0.544 | 0.673 | 0.857 | 0.597 | 0.690 | 0.526 | 0.721 | 0.626 |
| ESM2 + Aho hidden state fusion | 0.605 | 0.685 | 0.541 | 0.708 | 0.719 | 0.612 | **0.723** | 0.531 | 0.664 | 0.840 | 0.599 | 0.658 | 0.549 | 0.708 | 0.530 |
| ESM2 + Aho hidden state fusion only peptides | 0.581 | 0.665 | 0.515 | 0.696 | 0.615 | 0.584 | 0.718 | 0.491 | 0.652 | 0.828 | 0.578 | 0.628 | 0.536 | 0.704 | 0.357 |
| ESM2 + Aho сигнал добавляется к CRF переходам | 0.558 | 0.629 | 0.501 | N/A | N/A | 0.543 | 0.606 | 0.492 | N/A | N/A | 0.570 | 0.648 | 0.508 | N/A | N/A |
| ESM2 + Aho early fusion (concat with esm) | 0.594 | 0.633 | 0.560 | 0.738 | 0.651 | 0.568 | 0.603 | 0.537 | 0.672 | 0.836 | 0.616 | 0.659 | 0.578 | 0.708 | 0.412 |
| ESM2 + доп. лосс разрезов к ближайшей границе | 0.559 | 0.629 | 0.503 | N/A | N/A | 0.543 | 0.603 | 0.494 | N/A | N/A | 0.573 | 0.651 | 0.511 | N/A | N/A |
| ESM2 c AdamW оптимизатором | 0.560 | 0.602 | 0.524 | 0.726 | **0.811** | 0.541 | 0.560 | 0.524 | 0.687 | 0.883 | 0.577 | 0.642 | 0.524 | 0.702 | **0.660** |

**Footnotes — rows with N/A MCC/AUC:**

- *ESM2 + Aho сигнал добавляется к CRF переходам* (`esm2_aho_transition_bias_sparse_trainable_zero`): model unrecoverable for infer
- *ESM2 + доп. лосс разрезов к ближайшей границе* (`esm2_bond_loss_soft_l005_w5_tau15`): model unrecoverable for infer


## Table 2: Embedding Generators (TEST set)

| Model | All |  |  |  |  | Peptides |  |  |  |  | Propeptides |  |  |  |  |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ESM2 | 0.607 | 0.640 | 0.578 | 0.752 | 0.717 | **0.604** | 0.649 | 0.565 | 0.696 | 0.860 | 0.610 | 0.633 | **0.588** | **0.746** | 0.509 |
| ESM2+residue features (ESM2+ below) | 0.566 | 0.652 | 0.500 | 0.726 | 0.732 | 0.551 | 0.636 | 0.486 | 0.648 | 0.852 | 0.579 | 0.666 | 0.512 | 0.732 | 0.547 |
| ESM-C | 0.568 | 0.634 | 0.515 | N/A | N/A | 0.538 | 0.609 | 0.481 | N/A | N/A | 0.592 | 0.652 | 0.542 | N/A | N/A |
| ESM-C 6B | 0.579 | 0.570 | **0.590** | **0.759** | 0.766 | 0.562 | 0.526 | **0.605** | **0.729** | **0.886** | 0.595 | 0.613 | 0.577 | **0.746** | 0.587 |
| ProstT5 | 0.509 | 0.588 | 0.449 | 0.718 | **0.786** | 0.477 | 0.541 | 0.427 | 0.591 | 0.855 | 0.536 | 0.628 | 0.468 | 0.702 | 0.652 |
| ProstT5+residue features | 0.495 | 0.575 | 0.435 | 0.699 | 0.772 | 0.485 | 0.546 | 0.437 | 0.579 | 0.827 | 0.504 | 0.601 | 0.435 | 0.686 | **0.655** |
| (ProstT5 3DI + ESM2) proj. | 0.596 | 0.635 | 0.562 | N/A | N/A | 0.584 | 0.603 | 0.565 | N/A | N/A | 0.607 | 0.663 | 0.560 | N/A | N/A |
| (ProstT5 3DI + ESM2) proj.gated. | 0.545 | **0.697** | 0.447 | 0.659 | 0.395 | 0.530 | 0.612 | 0.468 | 0.604 | 0.565 | 0.559 | **0.798** | 0.430 | 0.682 | 0.228 |
| (ProstT5 3DI + ESM2) proj.gated.conv. | **0.611** | 0.658 | 0.571 | 0.731 | 0.476 | 0.603 | **0.664** | 0.552 | 0.682 | 0.685 | **0.618** | 0.653 | 0.587 | 0.708 | 0.235 |
| AFTK all, no filter | 0.382 | 0.529 | 0.299 | N/A | N/A | 0.382 | 0.464 | 0.324 | N/A | N/A | 0.382 | 0.611 | 0.278 | N/A | N/A |
| AFTK only single, no filter | 0.331 | 0.567 | 0.233 | N/A | N/A | 0.312 | 0.523 | 0.223 | N/A | N/A | 0.346 | 0.605 | 0.242 | N/A | N/A |
| AFTK all w/o lddt, no filter | 0.408 | 0.458 | 0.368 | N/A | N/A | 0.385 | 0.364 | 0.409 | N/A | N/A | 0.434 | 0.616 | 0.335 | N/A | N/A |
| AFTK all, >70% avg plddt | 0.274 | 0.316 | 0.242 | N/A | N/A | 0.069 | 0.065 | 0.074 | N/A | N/A | 0.397 | 0.531 | 0.317 | N/A | N/A |
| ESM2+(AFTK all, no filter) pr.gt.conv | 0.568 | 0.590 | 0.548 | N/A | N/A | 0.546 | 0.538 | 0.553 | N/A | N/A | 0.589 | 0.641 | 0.545 | N/A | N/A |
| ESM2+(AFTK only single no filter) pr.gt.conv | 0.577 | 0.612 | 0.545 | 0.723 | 0.613 | 0.594 | 0.612 | 0.578 | 0.687 | 0.812 | 0.562 | 0.612 | 0.519 | 0.679 | 0.368 |
| ESM2+(AFTK only pair no filter) pr.gt.conv | 0.595 | 0.651 | 0.547 | 0.721 | 0.498 | 0.576 | 0.606 | 0.548 | 0.677 | 0.736 | 0.611 | 0.694 | 0.546 | 0.697 | 0.242 |
| ESM2+(AFTK all w/o lddt no filter) pr.gt.conv | 0.565 | 0.585 | 0.546 | N/A | N/A | 0.546 | 0.546 | 0.545 | N/A | N/A | 0.582 | 0.622 | 0.547 | N/A | N/A |
| ESM2+(AFTK all, >70% avg plddt) pr.gt.conv | 0.574 | 0.593 | 0.555 | 0.662 | 0.476 | 0.524 | 0.564 | 0.489 | 0.567 | 0.839 | 0.595 | 0.605 | 0.585 | 0.703 | 0.233 |

**Footnotes — rows with N/A MCC/AUC:**

- *ESM-C* (`train_run_esmc_600m`): fresh infer diverged (drift=0.0151)
- *(ProstT5 3DI + ESM2) proj.* (`train_run_esm2+3di_proj`): fresh infer diverged (drift=0.0365)
- *AFTK all, no filter* (`train_run_aft`): fresh infer diverged (drift=0.0614)
- *AFTK only single, no filter* (`train_run_aft_single`): fresh infer diverged (drift=0.1129)
- *AFTK all w/o lddt, no filter* (`train_run_aft_no_lddt`): fresh infer diverged (drift=0.1391)
- *AFTK all, >70% avg plddt* (`train_run_aft_plddt70`): fresh infer diverged (drift=0.0339)
- *ESM2+(AFTK all, no filter) pr.gt.conv* (`train_run_esm2_aft`): fresh infer diverged (drift=0.0202)
- *ESM2+(AFTK all w/o lddt no filter) pr.gt.conv* (`train_run_esm2_aft_no_lddt_gated`): fresh infer diverged (drift=0.0491)


## Table 3: Homo sapiens Only (HOMO test set)

| Model | All |  |  |  |  | Peptides |  |  |  |  | Propeptides |  |  |  |  |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ESM2 | 0.460 | 0.487 | 0.435 | 0.697 | 0.608 | 0.339 | 0.370 | 0.312 | 0.576 | 0.884 | 0.529 | 0.551 | 0.509 | 0.743 | 0.428 |
| ESM2+residue features (ESM2+ below) | **0.514** | 0.638 | 0.430 | N/A | N/A | 0.385 | **0.526** | 0.303 | N/A | N/A | 0.587 | 0.692 | 0.509 | N/A | N/A |
| ESM-C | 0.477 | 0.587 | 0.402 | 0.677 | 0.630 | 0.208 | 0.333 | 0.152 | 0.561 | **0.895** | 0.598 | 0.667 | **0.542** | 0.672 | 0.477 |
| ProstT5 | 0.503 | 0.586 | 0.441 | N/A | N/A | 0.346 | 0.474 | 0.273 | N/A | N/A | 0.577 | 0.627 | 0.533 | N/A | N/A |
| ProstT5+residue features | 0.450 | 0.537 | 0.387 | 0.705 | **0.703** | 0.357 | 0.435 | 0.303 | 0.555 | 0.842 | 0.500 | 0.591 | 0.433 | 0.701 | **0.610** |
| (ProstT5 3DI + ESM2) proj. | 0.446 | 0.486 | 0.412 | N/A | N/A | 0.308 | 0.303 | 0.312 | N/A | N/A | 0.543 | 0.641 | 0.472 | N/A | N/A |
| (ProstT5 3DI + ESM2) proj. gated. | 0.508 | **0.717** | 0.393 | 0.613 | 0.331 | 0.320 | 0.421 | 0.258 | 0.474 | 0.506 | **0.625** | **0.926** | 0.472 | 0.658 | 0.220 |
| (ProstT5 3DI + ESM2) proj.gated.conv. | 0.483 | 0.571 | 0.419 | **0.726** | 0.424 | 0.286 | 0.348 | 0.242 | **0.628** | 0.726 | 0.602 | 0.700 | 0.528 | 0.720 | 0.241 |
| AFTK all, no filter | 0.256 | 0.432 | 0.182 | N/A | N/A | 0.091 | 0.167 | 0.062 | N/A | N/A | 0.346 | 0.560 | 0.250 | N/A | N/A |
| AFTK only single, no filter | 0.188 | 0.407 | 0.122 | N/A | N/A | 0.050 | 0.167 | 0.029 | N/A | N/A | 0.260 | 0.476 | 0.179 | N/A | N/A |
| AFTK all w/o lddt, no filter | 0.224 | 0.247 | 0.205 | N/A | N/A | 0.197 | 0.179 | 0.219 | N/A | N/A | 0.244 | 0.324 | 0.196 | N/A | N/A |
| AFTK all, >70% avg plddt | 0.400 | 0.536 | 0.319 | 0.678 | 0.529 | 0.000 | 0.000 | 0.000 | N/A | 0.773 | 0.441 | 0.536 | 0.375 | 0.689 | 0.510 |
| ESM2+(AFTK all, no filter) pr.gt.conv | 0.442 | 0.456 | 0.429 | N/A | N/A | 0.299 | 0.278 | 0.323 | N/A | N/A | 0.542 | 0.605 | 0.491 | N/A | N/A |
| ESM2+(AFTK only single, no filter) pr.gt.conv | 0.471 | 0.529 | 0.424 | 0.684 | 0.457 | **0.386** | 0.440 | **0.344** | 0.566 | 0.772 | 0.521 | 0.581 | 0.472 | 0.688 | 0.252 |
| ESM2+(AFTK only pair, no filter) pr.gt.conv | 0.446 | 0.585 | 0.360 | N/A | N/A | 0.291 | 0.364 | 0.242 | N/A | N/A | 0.548 | 0.742 | 0.434 | N/A | N/A |
| ESM2+(AFTK all w/o lddt, no filter) pr.gt.conv | 0.460 | 0.493 | 0.430 | N/A | N/A | 0.349 | 0.367 | 0.333 | N/A | N/A | 0.531 | 0.578 | 0.491 | N/A | N/A |
| ESM2+(AFTK all, >70% avg plddt) pr.gt.conv | 0.488 | 0.541 | **0.444** | 0.724 | 0.122 | 0.000 | 0.000 | 0.000 | -0.005 | 0.346 | 0.556 | 0.588 | 0.526 | **0.761** | 0.105 |

**Footnotes — rows with N/A MCC/AUC:**

- *ESM2+residue features (ESM2+ below)* (`train_run_esm2_plus`): fresh infer diverged (drift=0.0848)
- *ProstT5* (`train_run_prostt5`): fresh infer diverged (drift=0.0196)
- *(ProstT5 3DI + ESM2) proj.* (`train_run_esm2+3di_proj`): fresh infer diverged (drift=0.0625)
- *AFTK all, no filter* (`train_run_aft`): fresh infer diverged (drift=0.0629)
- *AFTK only single, no filter* (`train_run_aft_single`): fresh infer diverged (drift=0.0833)
- *AFTK all w/o lddt, no filter* (`train_run_aft_no_lddt`): fresh infer diverged (drift=0.3080)
- *ESM2+(AFTK all, no filter) pr.gt.conv* (`train_run_esm2_aft`): fresh infer diverged (drift=0.0365)
- *ESM2+(AFTK only pair, no filter) pr.gt.conv* (`train_run_esm2_aft_pair_gated`): fresh infer diverged (drift=0.0323)
- *ESM2+(AFTK all w/o lddt, no filter) pr.gt.conv* (`train_run_esm2_aft_no_lddt_gated`): fresh infer diverged (drift=0.0189)


## Coverage

The following run folders have `test_metrics.json` (included in `canonical_metrics.csv`) but are not mapped to any of the 3 experiment tables. They can be added in future tables.

- `esm2_aho_state_bias_pep_boundary_010` (test_drift=0.0000, trusted=True)
- `esm2_boundary_bond_l002_w5_tau15` (test_drift=0.0000, trusted=True)
- `esm2_boundary_only_scale10` (test_drift=0.0000, trusted=True)
- `esm2_lora_lstmcnncrf` (test_drift=0.0226, trusted=False)
- `train_run_esm2_100` (test_drift=0.0000, trusted=True)
- `train_run_esm2_25` (test_drift=0.0423, trusted=False)
- `train_run_esm2_50` (test_drift=0.0000, trusted=True)
- `train_run_esm2_75` (test_drift=0.0000, trusted=True)
- `train_run_esm2_conv` (test_drift=0.0000, trusted=True)
- `train_run_esm2_only_homo` (test_drift=0.0000, trusted=True)
- `train_run_esm2_plus_proj_gated` (test_drift=0.0000, trusted=True)
- `uni2026_run_esm2` (test_drift=0.0631, trusted=False)

The following run folders have `test_metrics_infer.json` but **no** `test_metrics.json`. They are excluded from the CSV entirely (no authoritative P/R/F1 source) and not in any table:

- `esm2_lora_lstmcnncrf_r4_last2_qv` (infer-only, no test_metrics.json)
- `train_run_3di_only` (infer-only, no test_metrics.json)

> **Dropped rows (no backing run folder):** `(ProstT5 3DI + ESM2+) proj.gated.conv.` was present in the LaTeX source for both Table 2 and Table 3 but has no matching run folder in `runs/`. These rows are omitted.
