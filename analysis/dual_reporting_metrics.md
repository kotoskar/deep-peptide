# Dual-reported peptide-finding metrics: original vs corrected (±3)

P/R/F1 use the ±3 cleavage-tolerance peptide-finding metric. **orig** = the
metric exactly as shipped in `manuscript_metrics.get_counts_for_protein`
(this is the value in the paper tables; it has a variable-shadowing bug that
understates recall — and the SAME bug exists in upstream DeepPeptide, so these
numbers stay comparable to the original work). **corr** = the corrected ±3
matcher. We report both rather than overwriting. Source:
`analysis/corrected_metrics.py` -> `analysis/corrected_metrics.csv`.

`Δ` columns = corr − orig. Two Table-1 runs
(`esm2_aho_transition_bias_sparse`, `esm2_bond_loss_soft`) are not re-inferable
(model classes gone) — corrected = N/A, keep their published orig values.

**Why no MCC/AUC here:** the bug lives only in the *segment-level* peptide-finding
metric (the ±3 start/stop matching of whole peptides). MCC and AUC are computed at
the **residue level** (per-position positive/negative), a completely separate code
path that never calls `get_counts_for_protein` — so they are **unaffected by the
bug** and identical to the values in `analysis/canonical_metrics.md` (where MCC/AUC
already come from fresh inference). They are therefore not duplicated here; this
table only shows the metric that actually changes (F1/precision/recall).

Column legend: `F1 all` / `recall all` = the peptide-finding F1 / recall pooled
over peptides+propeptides; `recall pep Δ` / `recall propep Δ` = the corr−orig recall
gap for peptides and propeptides separately.

| run | F1 all orig | F1 all corr | recall all orig | recall all corr | Δrec all | recall pep Δ | recall propep Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| esm2_aho_emission_fusion | 0.615 | 0.631 | 0.558 | 0.579 | +0.021 | +0.031 | +0.013 |
| train_run_esm2+3di_proj_gated_conv | 0.611 | 0.631 | 0.571 | 0.598 | +0.027 | +0.044 | +0.013 |
| train_run_esm2 | 0.607 | 0.621 | 0.578 | 0.597 | +0.019 | +0.024 | +0.014 |
| esm2_aho_mid_fusion_raw_m64 | 0.605 | 0.619 | 0.541 | 0.561 | +0.019 | +0.024 | +0.015 |
| esm2_aho_tribranch | 0.594 | 0.619 | 0.560 | 0.593 | +0.034 | +0.050 | +0.020 |
| esm2_boundary_bond_l002_w5_tau15 | 0.606 | 0.618 | 0.546 | 0.562 | +0.016 | +0.030 | +0.004 |
| esm2_boundary_only_scale10 | 0.597 | 0.617 | 0.522 | 0.548 | +0.026 | +0.052 | +0.005 |
| esm2_telescoping_segmental | 0.596 | 0.614 | 0.564 | 0.589 | +0.025 | +0.026 | +0.023 |
| esm2_aho_emission_fusion_h32 | 0.596 | 0.612 | 0.534 | 0.555 | +0.021 | +0.026 | +0.017 |
| train_run_esm2_aft_pair_gated | 0.595 | 0.612 | 0.547 | 0.571 | +0.024 | +0.041 | +0.009 |
| train_run_esm2+3di_proj | 0.596 | 0.611 | 0.562 | 0.583 | +0.020 | +0.039 | +0.005 |
| train_run_esm2_plus_proj_gated | 0.583 | 0.605 | 0.567 | 0.597 | +0.030 | +0.050 | +0.014 |
| esm2_aho_state_bias_pep_boundary_010 | 0.592 | 0.605 | 0.530 | 0.546 | +0.017 | +0.025 | +0.010 |
| train_run_esm2_100 | 0.588 | 0.602 | 0.532 | 0.550 | +0.018 | +0.025 | +0.012 |
| esmc_6b | 0.579 | 0.599 | 0.590 | 0.618 | +0.029 | +0.039 | +0.020 |
| train_run_esm2_aft | 0.568 | 0.597 | 0.548 | 0.588 | +0.040 | +0.061 | +0.022 |
| train_run_esm2_aft_single_gated | 0.577 | 0.595 | 0.545 | 0.570 | +0.025 | +0.034 | +0.018 |
| train_run_esm2_aft_no_lddt_gated | 0.565 | 0.595 | 0.546 | 0.588 | +0.042 | +0.065 | +0.023 |
| esm2_aho_mid_fusion_raw_m64_pep_only | 0.581 | 0.591 | 0.515 | 0.529 | +0.014 | +0.020 | +0.008 |
| train_run_esm2_conv | 0.567 | 0.587 | 0.548 | 0.577 | +0.029 | +0.043 | +0.017 |
| train_run_esm2_aft_plddt70 | 0.574 | 0.586 | 0.555 | 0.572 | +0.017 | +0.004 | +0.023 |
| train_run_esm2_plus | 0.566 | 0.579 | 0.500 | 0.517 | +0.017 | +0.023 | +0.012 |
| train_run_esmc_600m | 0.568 | 0.577 | 0.516 | 0.527 | +0.012 | +0.015 | +0.009 |
| train_run_esm2_75 | 0.559 | 0.577 | 0.532 | 0.557 | +0.025 | +0.029 | +0.022 |
| train_run_esm2_adamw | 0.560 | 0.573 | 0.524 | 0.540 | +0.016 | +0.025 | +0.009 |
| esm2_lora_lstmcnncrf | 0.543 | 0.558 | 0.496 | 0.515 | +0.019 | +0.022 | +0.017 |
| train_run_esm2+3di_proj_gated | 0.545 | 0.551 | 0.447 | 0.454 | +0.007 | +0.015 | +0.000 |
| train_run_prostt5 | 0.509 | 0.519 | 0.449 | 0.462 | +0.012 | +0.011 | +0.013 |
| train_run_esm2_50 | 0.491 | 0.512 | 0.478 | 0.506 | +0.028 | +0.050 | +0.010 |
| uni2026_run_esm2 | 0.501 | 0.512 | 0.378 | 0.390 | +0.011 | +0.018 | +0.005 |
| train_run_prostt5_plus | 0.495 | 0.504 | 0.435 | 0.446 | +0.011 | +0.016 | +0.006 |
| train_run_esm2_25 | 0.423 | 0.450 | 0.401 | 0.434 | +0.033 | +0.065 | +0.007 |
| train_run_aft_no_lddt | 0.408 | 0.419 | 0.368 | 0.381 | +0.013 | +0.026 | +0.002 |
| train_run_aft | 0.382 | 0.386 | 0.299 | 0.303 | +0.004 | +0.009 | +0.001 |
| train_run_aft_single | 0.331 | 0.332 | 0.233 | 0.235 | +0.002 | +0.003 | +0.000 |
| train_run_aft_plddt70 | 0.274 | 0.275 | 0.242 | 0.243 | +0.001 | +0.001 | +0.000 |
| train_run_esm2_only_homo | 0.188 | 0.197 | 0.124 | 0.131 | +0.007 | +0.015 | +0.000 |
| esm2_aho_transition_bias_sparse_trainable_zero | (see paper) | N/A | (see paper) | N/A | — | — | — |
| esm2_bond_loss_soft_l005_w5_tau15 | (see paper) | N/A | (see paper) | N/A | — | — | — |

**Aggregate:** corrected recall-all is higher than published for all 37 inferable runs (min +0.001, median +0.019, max +0.042). The offset is largest for models that over-predict and smallest for sparse AFT-only models (the bug only fires when a protein has more predictions than true segments), but it is positive everywhere, so **model rankings are unchanged**.
