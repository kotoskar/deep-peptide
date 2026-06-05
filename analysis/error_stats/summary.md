# Error analysis (±3 cleavage-tolerance matching)

Runs (gate-passed): esm2_aho_emission_fusion, esm2_aho_emission_fusion_h32, esm2_aho_mid_fusion_raw_m64, esm2_telescoping_segmental, train_run_esm2, train_run_esm2+3di_proj, train_run_esm2+3di_proj_gated_conv, train_run_esm2_aft_single_gated, train_run_esmc_600m

Matching = manuscript ±3 (a true peptide group is recovered iff some prediction's start AND stop are within ±3). Recall = TP/(TP+FN) over true groups; FP = predicted segments matching no true group.

## Recall by length bin — peptides (pooled over gate-passed runs)

| length | n true | recall | FN |
|---|---:|---:|---:|
| 5 | 162 | 0.358 | 104 |
| 6-10 | 1166 | 0.746 | 296 |
| 11-20 | 2075 | 0.694 | 635 |
| 21-30 | 4179 | 0.602 | 1664 |
| 31-50 | 2661 | 0.376 | 1660 |
| 51+ | 0 | nan | 0 |

## Recall by length bin — propeptides (pooled over gate-passed runs)

| length | n true | recall | FN |
|---|---:|---:|---:|
| 5 | 252 | 0.349 | 164 |
| 6-10 | 2599 | 0.289 | 1848 |
| 11-20 | 2364 | 0.574 | 1006 |
| 21-30 | 4725 | 0.790 | 993 |
| 31-50 | 2540 | 0.477 | 1328 |
| 51+ | 0 | nan | 0 |

## Tiny peptides (length = 5)

- **peptides**: 162 true len-5 segments, recall=0.358; len-5 FN = 104 of 4359 total FN (2.4%).
- **propeptides**: 252 true len-5 segments, recall=0.349; len-5 FN = 164 of 5339 total FN (3.1%).

## Recall by organism — peptides (top 12 by true count, pooled)

| organism | n true | recall |
|---|---:|---:|
| Bombyx mori | 549 | 0.896 |
| Cyriopagopus hainanus | 495 | 0.046 |
| Caenorhabditis elegans | 495 | 0.428 |
| Homo sapiens | 270 | 0.381 |
| Procambarus clarkii | 261 | 0.897 |
| Rattus norvegicus | 243 | 0.469 |
| Agrotis ipsilon | 216 | 0.764 |
| Mus musculus | 207 | 0.531 |
| Bos taurus | 204 | 0.348 |
| Conus textile | 198 | 0.793 |
| Drosophila melanogaster | 171 | 0.749 |
| Aplysia californica | 162 | 0.272 |

## Per-run recall (peptides / propeptides) and validation gate

| run | recall pep | recall propep | gate max|Δ| |
|---|---:|---:|---:|
| esm2_aho_emission_fusion | 0.560 | 0.595 | 3.1e-02 |
| esm2_aho_emission_fusion_h32 | 0.570 | 0.544 | 2.6e-02 |
| esm2_aho_mid_fusion_raw_m64 | 0.556 | 0.565 | 2.4e-02 |
| esm2_telescoping_segmental | 0.589 | 0.589 | 2.6e-02 |
| train_run_esm2 | 0.590 | 0.602 | 2.4e-02 |
| train_run_esm2+3di_proj | 0.604 | 0.565 | 3.9e-02 |
| train_run_esm2+3di_proj_gated_conv | 0.596 | 0.600 | 4.4e-02 |
| train_run_esm2_aft_single_gated | 0.612 | 0.536 | 3.4e-02 |
| train_run_esmc_600m | 0.495 | 0.553 | 1.4e-02 |

## Finding: corrected ±3 matching vs published per-peptide metric

`manuscript_metrics.get_counts_for_protein` has a variable-shadowing bug (the inner `for idx, row in pred_df.iterrows()` reuses `idx`, so `true_df.loc[idx,'matched']=True` writes the matched flag to the row whose label equals the *pred* index, not the true row). It mostly cancels in the aggregate but diverges when a protein has more predictions than true segments. Below: published recall (buggy) vs this correct ±3 matcher.

| run | pep recall pub | pep recall correct | Δ | propep recall pub | propep recall correct | Δ |
|---|---:|---:|---:|---:|---:|---:|
| esm2_aho_emission_fusion | 0.529 | 0.560 | +0.031 | 0.582 | 0.595 | +0.013 |
| esm2_aho_emission_fusion_h32 | 0.544 | 0.570 | +0.026 | 0.526 | 0.544 | +0.017 |
| esm2_aho_mid_fusion_raw_m64 | 0.531 | 0.556 | +0.024 | 0.549 | 0.565 | +0.015 |
| esm2_telescoping_segmental | 0.562 | 0.589 | +0.026 | 0.566 | 0.589 | +0.023 |
| train_run_esm2 | 0.565 | 0.590 | +0.024 | 0.588 | 0.602 | +0.014 |
| train_run_esm2+3di_proj | 0.565 | 0.604 | +0.039 | 0.560 | 0.565 | +0.005 |
| train_run_esm2+3di_proj_gated_conv | 0.552 | 0.596 | +0.044 | 0.587 | 0.600 | +0.013 |
| train_run_esm2_aft_single_gated | 0.578 | 0.612 | +0.034 | 0.519 | 0.536 | +0.018 |
| train_run_esmc_600m | 0.481 | 0.495 | +0.014 | 0.542 | 0.553 | +0.011 |
