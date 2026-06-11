# Canonical Experiment Tables

> ⚠️ **Metric note:** the P/R/F1 here come from the shipped ±3 peptide-finding
> metric, which has a variable-shadowing bug (inherited from upstream DeepPeptide)
> that understates recall by ~2–4 pp. These values are kept for comparability with
> the paper. See `analysis/dual_reporting_metrics.md` for the original-vs-corrected
> table, and `texs/error_analysis/report.md` §4 for the bug writeup.

> **Methodology:** P/R/F1 values are authoritative train-time values from `test_metrics.json` (or `homo_test_metrics.json` for Table 3). MCC and AUC are from fresh fp32 inference (`test_metrics_infer.json` / `homo_test_metrics_infer.json`), accepted only when `drift = max|train-time P/R/F1 − fresh P/R/F1|` ≤ 0.015. Hard overrides (always N/A): `esm2_bond_loss_soft_l005_w5_tau15` and `esm2_aho_transition_bias_sparse_trainable_zero` (model unrecoverable for infer). Values rounded to 3 decimal places. **Bold** = best in column (N/A cells excluded).

## Headline: best combined configuration (architecture × embedding)

Not part of the original Table 1/2 sweep — this pairs the best **embedding** (ESM-C 6B, top residue-level signal) with a boundary-sharpening **architecture** (`lstmcnncrf_boundary_bond_loss`). It is the best F1 and MCC in the project; the boundary head turns ESM-C 6B's high-recall/low-precision signal into precision (+0.14). See `texs/error_analysis/combine_best.md`. (Single seed; deterministic fp32, drift 0.000.)

| Config | TEST F1 all | TEST Prec all | TEST Rec all | TEST MCC all | HOMO F1 all | HOMO MCC all |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ESM-C 6B + boundary/bond** | 0.657 | 0.714 | 0.608 | 0.765 | 0.548 | 0.747 |
| ESM2 baseline | 0.607 | 0.640 | 0.578 | 0.750 | 0.460 | 0.693 |
| ESM-C 6B baseline | 0.579 | 0.570 | 0.590 | 0.758 | 0.476 | 0.686 |

## Table 1: Architectural Changes (TEST set)

| Model | All F1 | All Prec | All Rec | All MCC | All AUC | Pep F1 | Pep Prec | Pep Rec | Pep MCC | Pep AUC | Propep F1 | Propep Prec | Propep Rec | Propep MCC | Propep AUC |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ESM2 (baseline) | 0.607 | 0.640 | **0.578** | **0.750** | **0.963** | 0.604 | 0.649 | **0.565** | **0.694** | **0.969** | 0.610 | 0.633 | **0.588** | **0.745** | 0.963 |
| ESM2 + telescopic CRF | 0.596 | 0.632 | 0.564 | 0.731 | 0.949 | **0.614** | 0.675 | 0.562 | 0.679 | 0.958 | 0.582 | 0.600 | 0.566 | 0.706 | 0.954 |
| ESM2 + Aho emission fusion | **0.615** | **0.686** | 0.558 | 0.737 | 0.949 | 0.594 | 0.676 | 0.529 | 0.681 | 0.958 | **0.633** | **0.693** | 0.582 | 0.735 | 0.954 |
| ESM2 + (Aho -> hidden layer 32) emission fusion | 0.596 | 0.674 | 0.534 | 0.733 | 0.959 | 0.595 | 0.657 | 0.544 | 0.673 | 0.960 | 0.597 | 0.690 | 0.526 | 0.721 | **0.966** |
| ESM2 + Aho hidden state fusion | 0.605 | 0.685 | 0.541 | 0.707 | 0.946 | 0.612 | **0.723** | 0.531 | 0.663 | 0.946 | 0.599 | 0.658 | 0.549 | 0.708 | 0.958 |
| ESM2 + Aho hidden state fusion only peptides | 0.581 | 0.665 | 0.515 | 0.696 | 0.932 | 0.584 | 0.718 | 0.491 | 0.650 | 0.941 | 0.578 | 0.628 | 0.536 | 0.704 | 0.945 |
| ESM2 + Aho сигнал добавляется к CRF переходам | 0.558 | 0.629 | 0.501 | N/A | N/A | 0.543 | 0.606 | 0.492 | N/A | N/A | 0.570 | 0.648 | 0.508 | N/A | N/A |
| ESM2 + Aho early fusion (concat with esm) | 0.594 | 0.633 | 0.560 | 0.738 | 0.960 | 0.568 | 0.603 | 0.537 | 0.672 | 0.958 | 0.616 | 0.659 | 0.578 | 0.708 | 0.961 |
| ESM2 + доп. лосс разрезов к ближайшей границе | 0.559 | 0.629 | 0.503 | N/A | N/A | 0.543 | 0.603 | 0.494 | N/A | N/A | 0.573 | 0.651 | 0.511 | N/A | N/A |
| ESM2 c AdamW оптимизатором | 0.560 | 0.602 | 0.524 | 0.729 | 0.962 | 0.541 | 0.560 | 0.524 | 0.687 | 0.968 | 0.577 | 0.642 | 0.524 | 0.703 | 0.966 |

**Footnotes — rows with N/A MCC/AUC:**

- *ESM2 + Aho сигнал добавляется к CRF переходам* (`esm2_aho_transition_bias_sparse_trainable_zero`): model unrecoverable for infer
- *ESM2 + доп. лосс разрезов к ближайшей границе* (`esm2_bond_loss_soft_l005_w5_tau15`): model unrecoverable for infer


## Table 2: Embedding Generators (TEST set)

| Model | All F1 | All Prec | All Rec | All MCC | All AUC | Pep F1 | Pep Prec | Pep Rec | Pep MCC | Pep AUC | Propep F1 | Propep Prec | Propep Rec | Propep MCC | Propep AUC |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ESM2 | 0.607 | 0.640 | 0.578 | 0.750 | 0.963 | **0.604** | 0.649 | 0.565 | 0.694 | 0.969 | 0.610 | 0.633 | **0.588** | 0.745 | 0.963 |
| ESM2+residue features (ESM2+ below) | 0.566 | 0.652 | 0.500 | 0.730 | 0.948 | 0.551 | 0.636 | 0.486 | 0.653 | 0.951 | 0.579 | 0.666 | 0.512 | 0.735 | 0.961 |
| ESM-C | 0.568 | 0.634 | 0.515 | 0.713 | 0.954 | 0.538 | 0.609 | 0.481 | 0.629 | 0.958 | 0.592 | 0.652 | 0.542 | 0.706 | 0.960 |
| ESM-C 6B | 0.579 | 0.570 | **0.590** | **0.758** | **0.968** | 0.562 | 0.526 | **0.605** | **0.728** | **0.971** | 0.595 | 0.613 | 0.577 | **0.746** | **0.971** |
| ProstT5 | 0.509 | 0.588 | 0.449 | 0.719 | 0.957 | 0.477 | 0.541 | 0.427 | 0.591 | 0.945 | 0.536 | 0.628 | 0.468 | 0.704 | 0.966 |
| ProstT5+residue features | 0.495 | 0.575 | 0.435 | 0.697 | 0.943 | 0.485 | 0.546 | 0.437 | 0.575 | 0.916 | 0.504 | 0.601 | 0.435 | 0.685 | 0.968 |
| (ProstT5 3DI + ESM2) proj. | 0.596 | 0.635 | 0.562 | 0.745 | 0.919 | 0.584 | 0.603 | 0.565 | 0.698 | 0.934 | 0.607 | 0.663 | 0.560 | 0.711 | 0.920 |
| (ProstT5 3DI + ESM2) proj.gated. | 0.545 | **0.697** | 0.447 | 0.659 | 0.807 | 0.530 | 0.612 | 0.468 | 0.604 | 0.838 | 0.559 | **0.798** | 0.430 | 0.682 | 0.806 |
| (ProstT5 3DI + ESM2) proj.gated.conv. | **0.611** | 0.658 | 0.571 | 0.731 | 0.912 | 0.603 | **0.664** | 0.552 | 0.682 | 0.916 | **0.618** | 0.653 | 0.587 | 0.708 | 0.908 |
| AFTK all, no filter | 0.382 | 0.529 | 0.299 | 0.550 | 0.946 | 0.382 | 0.464 | 0.324 | 0.498 | 0.947 | 0.382 | 0.611 | 0.278 | 0.522 | 0.952 |
| AFTK only single, no filter | 0.331 | 0.567 | 0.233 | 0.531 | 0.939 | 0.312 | 0.523 | 0.223 | 0.468 | 0.934 | 0.346 | 0.605 | 0.242 | 0.525 | 0.945 |
| AFTK all w/o lddt, no filter | 0.408 | 0.458 | 0.368 | 0.615 | 0.943 | 0.385 | 0.364 | 0.409 | 0.544 | 0.944 | 0.434 | 0.616 | 0.335 | 0.528 | 0.930 |
| AFTK all, >70% avg plddt | 0.274 | 0.316 | 0.242 | 0.579 | 0.950 | 0.069 | 0.065 | 0.074 | 0.483 | 0.952 | 0.397 | 0.531 | 0.317 | 0.517 | 0.945 |
| ESM2+(AFTK all, no filter) pr.gt.conv | 0.568 | 0.590 | 0.548 | 0.732 | 0.954 | 0.546 | 0.538 | 0.553 | 0.663 | 0.957 | 0.589 | 0.641 | 0.545 | 0.688 | 0.950 |
| ESM2+(AFTK only single no filter) pr.gt.conv | 0.577 | 0.612 | 0.545 | 0.722 | 0.933 | 0.594 | 0.612 | 0.578 | 0.685 | 0.937 | 0.562 | 0.612 | 0.519 | 0.674 | 0.935 |
| ESM2+(AFTK only pair no filter) pr.gt.conv | 0.595 | 0.651 | 0.547 | 0.722 | 0.871 | 0.576 | 0.606 | 0.548 | 0.680 | 0.868 | 0.611 | 0.694 | 0.546 | 0.700 | 0.882 |
| ESM2+(AFTK all w/o lddt no filter) pr.gt.conv | 0.565 | 0.585 | 0.546 | 0.744 | 0.959 | 0.546 | 0.546 | 0.545 | 0.692 | 0.958 | 0.582 | 0.622 | 0.547 | 0.693 | 0.952 |
| ESM2+(AFTK all, >70% avg plddt) pr.gt.conv | 0.574 | 0.593 | 0.555 | 0.662 | 0.937 | 0.524 | 0.564 | 0.489 | 0.567 | 0.919 | 0.595 | 0.605 | 0.585 | 0.702 | 0.964 |

**Footnotes — rows with N/A MCC/AUC:**

*(none — all rows trusted)*


## Table 3: Homo sapiens Only (HOMO test set)

| Model | All F1 | All Prec | All Rec | All MCC | All AUC | Pep F1 | Pep Prec | Pep Rec | Pep MCC | Pep AUC | Propep F1 | Propep Prec | Propep Rec | Propep MCC | Propep AUC |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ESM2 | 0.460 | 0.487 | 0.435 | 0.693 | 0.947 | 0.339 | 0.370 | 0.312 | 0.552 | **0.946** | 0.529 | 0.551 | 0.509 | 0.744 | 0.960 |
| ESM2+residue features (ESM2+ below) | **0.514** | 0.638 | 0.430 | 0.715 | 0.940 | 0.385 | **0.526** | 0.303 | **0.619** | 0.929 | 0.587 | 0.692 | 0.509 | 0.730 | 0.953 |
| ESM-C | 0.477 | 0.587 | 0.402 | 0.679 | 0.939 | 0.208 | 0.333 | 0.152 | 0.562 | 0.939 | 0.598 | 0.667 | **0.542** | 0.682 | 0.941 |
| ProstT5 | 0.503 | 0.586 | 0.441 | 0.695 | 0.952 | 0.346 | 0.474 | 0.273 | 0.589 | 0.909 | 0.577 | 0.627 | 0.533 | 0.716 | 0.968 |
| ProstT5+residue features | 0.450 | 0.537 | 0.387 | 0.709 | 0.959 | 0.357 | 0.435 | 0.303 | 0.566 | 0.901 | 0.500 | 0.591 | 0.433 | 0.701 | **0.973** |
| (ProstT5 3DI + ESM2) proj. | 0.446 | 0.486 | 0.412 | 0.628 | 0.916 | 0.308 | 0.303 | 0.312 | 0.519 | 0.897 | 0.543 | 0.641 | 0.472 | 0.646 | 0.876 |
| (ProstT5 3DI + ESM2) proj. gated. | 0.508 | **0.717** | 0.393 | 0.611 | 0.717 | 0.320 | 0.421 | 0.258 | 0.468 | 0.755 | **0.625** | **0.926** | 0.472 | 0.658 | 0.683 |
| (ProstT5 3DI + ESM2) proj.gated.conv. | 0.483 | 0.571 | 0.419 | 0.724 | 0.901 | 0.286 | 0.348 | 0.242 | 0.610 | 0.873 | 0.602 | 0.700 | 0.528 | 0.720 | 0.904 |
| AFTK all, no filter | 0.256 | 0.432 | 0.182 | 0.491 | 0.952 | 0.091 | 0.167 | 0.062 | 0.312 | 0.929 | 0.346 | 0.560 | 0.250 | 0.553 | 0.971 |
| AFTK only single, no filter | 0.188 | 0.407 | 0.122 | 0.476 | 0.947 | 0.050 | 0.167 | 0.029 | 0.255 | 0.941 | 0.260 | 0.476 | 0.179 | 0.528 | 0.969 |
| AFTK all w/o lddt, no filter | 0.224 | 0.247 | 0.205 | 0.521 | 0.947 | 0.197 | 0.179 | 0.219 | 0.422 | 0.923 | 0.244 | 0.324 | 0.196 | 0.531 | 0.955 |
| AFTK all, >70% avg plddt | 0.400 | 0.536 | 0.319 | 0.682 | **0.973** | 0.000 | 0.000 | 0.000 | N/A | 0.857 | 0.441 | 0.536 | 0.375 | 0.700 | 0.970 |
| ESM2+(AFTK all, no filter) pr.gt.conv | 0.442 | 0.456 | 0.429 | 0.663 | 0.948 | 0.299 | 0.278 | 0.323 | 0.531 | 0.927 | 0.542 | 0.605 | 0.491 | 0.697 | 0.929 |
| ESM2+(AFTK only single, no filter) pr.gt.conv | 0.471 | 0.529 | 0.424 | 0.692 | 0.891 | **0.386** | 0.440 | **0.344** | 0.581 | 0.878 | 0.521 | 0.581 | 0.472 | 0.690 | 0.877 |
| ESM2+(AFTK only pair, no filter) pr.gt.conv | 0.446 | 0.585 | 0.360 | 0.678 | 0.784 | 0.291 | 0.364 | 0.242 | 0.614 | 0.828 | 0.548 | 0.742 | 0.434 | 0.677 | 0.768 |
| ESM2+(AFTK all w/o lddt, no filter) pr.gt.conv | 0.460 | 0.493 | 0.430 | 0.685 | 0.934 | 0.349 | 0.367 | 0.333 | 0.543 | 0.911 | 0.531 | 0.578 | 0.491 | 0.710 | 0.930 |
| ESM2+(AFTK all, >70% avg plddt) pr.gt.conv | 0.488 | 0.541 | **0.444** | **0.732** | 0.933 | 0.000 | 0.000 | 0.000 | -0.004 | 0.381 | 0.556 | 0.588 | 0.526 | **0.764** | 0.972 |

**Footnotes — rows with N/A MCC/AUC:**

- *AFTK all, >70% avg plddt* (`train_run_aft_plddt70`): MCC undefined (no positive predictions for that class)


## Coverage

The following run folders have `test_metrics.json` (included in `canonical_metrics.csv`) but are not mapped to any of the 3 experiment tables. They can be added in future tables.

- `esm2_aho_state_bias_pep_boundary_010` (test_drift=0.0000, trusted=True)
- `esm2_boundary_bond_l002_w5_tau15` (test_drift=0.0000, trusted=True)
- `esm2_boundary_only_scale10` (test_drift=0.0000, trusted=True)
- `esm2_lora_lstmcnncrf` (test_drift=0.0000, trusted=True)
- `esmc6b_boundary_bond` (test_drift=0.0000, trusted=True)
- `esmc6b_telescoping` (test_drift=0.0000, trusted=True)
- `scaling_trainfrac50` (test_drift=0.0000, trusted=True)
- `scaling_trainfrac60` (test_drift=0.0000, trusted=True)
- `scaling_trainfrac70` (test_drift=0.0000, trusted=True)
- `scaling_trainfrac80` (test_drift=0.0000, trusted=True)
- `scaling_trainfrac90` (test_drift=0.0000, trusted=True)
- `train_run_esm2_100` (test_drift=0.0000, trusted=True)
- `train_run_esm2_25` (test_drift=0.0423, trusted=False)
- `train_run_esm2_50` (test_drift=0.0000, trusted=True)
- `train_run_esm2_75` (test_drift=0.0000, trusted=True)
- `train_run_esm2_conv` (test_drift=0.0000, trusted=True)
- `train_run_esm2_only_homo` (test_drift=0.0000, trusted=True)
- `train_run_esm2_plus_proj_gated` (test_drift=0.0000, trusted=True)
- `uni2026_run_esm2` (test_drift=0.0000, trusted=True)

The following run folders have `test_metrics_infer.json` but **no** `test_metrics.json`. They are excluded from the CSV entirely (no authoritative P/R/F1 source) and not in any table:

- `esm2_lora_lstmcnncrf_r4_last2_qv` (infer-only, no test_metrics.json)
- `train_run_3di_only` (infer-only, no test_metrics.json)

> **Dropped rows (no backing run folder):** `(ProstT5 3DI + ESM2+) proj.gated.conv.` was present in the LaTeX source for both Table 2 and Table 3 but has no matching run folder in `runs/`. These rows are omitted.
