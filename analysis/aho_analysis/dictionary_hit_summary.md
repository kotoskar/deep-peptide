# AHO refinement: stratify by whether the DICTIONARY actually fires

Per true TEST peptide, `aho_hit` = the precomputed `pep.inside` AHO feature is nonzero somewhere in the peptide span (some dictionary peptide overlaps it). `hit_source` distinguishes a train(uniprot) hit from an external-AMP-DB-only hit. Recall uplift = AHO models (mean) − baseline, within each bucket.

**Dictionary coverage of true test peptides:** 19.8% have a hit. Hit-source breakdown: {'external_only': 207, 'train_only': 19}

| bucket | n | baseline recall | AHO recall (mean) | uplift |
|---|---:|---:|---:|---:|
| dictionary HIT | 226 | 0.668 | 0.748 | +0.080 |
|   ↳ train hit | 19 | 0.842 | 0.895 | +0.053 |
|   ↳ external-only hit | 207 | 0.652 | 0.734 | +0.082 |
| NO hit | 915 | 0.570 | 0.516 | -0.055 |

**Reading:** the no-hit bucket isolates peptides the dictionary cannot help (no signal) — uplift there should be ~0/negative (AHO channel = noise). The hit buckets show whether AHO actually helps where it fires, and whether an external-DB hit (peptide novel to train but known to an AMP DB) is exploited as well as a train hit.