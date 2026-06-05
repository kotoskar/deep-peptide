# AHO prior: benefit on similar-to-train vs novel test peptides

Recall (±3) on true TEST **peptides**, stratified by max identity to any train peptide. similar = ≥70% identity, novel = <70%. AHO dictionary is fold-aware (train folds + external AMP DBs), so it can only match peptides resembling known ones.

Bucket sizes (true test peptides): {'novel(<70%)': 1066, 'similar(≥70%)': 73, 'unknown': 2}

| model | recall novel(<70%) | recall similar(≥70%) | recall all | Δ(sim−novel) |
|---|---:|---:|---:|---:|
| baseline | 0.571 | 0.849 | 0.590 | +0.278 |
| esm2_aho_emission_fusion | 0.537 | 0.890 | 0.560 | +0.354 |
| esm2_aho_emission_fusion_h32 | 0.545 | 0.932 | 0.570 | +0.386 |
| esm2_aho_mid_fusion_raw_m64 | 0.532 | 0.890 | 0.556 | +0.359 |

## AHO uplift over baseline, per similarity bucket

| bucket | baseline recall | AHO recall (mean of models) | uplift |
|---|---:|---:|---:|
| novel(<70%) | 0.571 | 0.538 | -0.033 |
| similar(≥70%) | 0.849 | 0.904 | +0.055 |

**Reading:** if the AHO uplift is concentrated in the similar(≥70%) bucket and ~0 (or negative) on novel peptides, the AHO prior works by retrieving known peptides rather than improving genuine generalization — quantifying the supervisor's concern.