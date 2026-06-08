# Where the errors come from — and what "more data" actually means

A natural hypothesis for the data-ceiling argument is: *the model fails on organisms
that are under-represented in training.* We tested it directly, and it is **wrong** —
in an informative way. The correct axis is coverage of **peptide sequence space**, not
organism headcount.

## 1. Organism headcount does NOT predict error (it slightly anti-correlates)

For each organism: number of peptide segments in TRAIN vs the baseline model's TEST
peptide recall (`analysis/error_vs_train_abundance.py`).

![Recall vs organism train abundance](figures/error_vs_train_abundance.png)

Spearman ρ(train count, recall) = **−0.70**. Examples:

| organism | train peptides | test recall |
|---|---:|---:|
| Procambarus clarkii | 1 | 0.93 |
| Bombyx mori | 13 | 0.92 |
| Cyriopagopus hainanus | 48 | 0.07 |
| Homo sapiens | 153 | 0.43 |
| Aplysia californica | 192 | 0.33 |

An organism with a **single** training peptide (Procambarus) is recovered at 0.93,
while *Homo sapiens* (153 train peptides) sits at 0.43. So "few examples of this
organism" is **not** the cause of failure.

**Why:** the model learns peptide *families*, not organisms. Conserved toxin /
neuropeptide families recur across species, so a barely-represented organism is
recovered when its peptides belong to a well-covered family. Conversely, well-studied
mammals contribute many *diverse* peptides, and the GraphPart split (held-out proteins
<30% identical) makes their TEST peptides genuinely novel — the hard ones.

## 2. Peptide-level training coverage DOES predict error

Bin each TEST peptide by its maximum identity to any TRAIN peptide
(`analysis/aho_analysis/aho_segments.csv`, baseline ESM2):

![Recall vs peptide coverage](figures/recall_vs_peptide_coverage.png)

| max identity to a train peptide | n | recall |
|---|---:|---:|
| < 0.30 (no similar train peptide) | 121 | **0.39** |
| 0.30–0.40 | 478 | 0.57 |
| 0.40–0.50 | 266 | 0.62 |
| 0.50–0.60 | 145 | 0.61 |
| 0.60–0.70 | 56 | 0.63 |
| ≥ 0.70 (close train match) | 73 | **0.85** |

Recall rises monotonically with how well the *peptide* is covered in training: peptides
with no similar training example are recovered at 0.39; those with a ≥70%-identical
training peptide at 0.85.

## 3. The corrected "more data" argument

Putting the three pieces together:

- **Data-scaling curve still rising at 100%** (`data_scaling.md`): more training data
  keeps improving the model — not yet at the ceiling.
- **Recall is governed by peptide-family coverage** (§2), not organism headcount (§1).
- Held-out peptides are **mostly novel** (only ~6% are ≥70% similar to train;
  `peptide_similarity.md`).

So the lever is **more data that broadens coverage of peptide sequence space** — new
peptide families, not more proteins from families already seen. Adding 1000 more human
proteins whose peptides resemble existing training data would help little; sampling
under-covered peptide families (the <0.30-similarity tail, where recall is 0.39) is
what would move the needle. This is a sharper, more actionable version of "we need more
data" than a raw per-organism count would suggest.
