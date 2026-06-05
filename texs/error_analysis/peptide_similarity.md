# Peptide-level similarity to train, and its link to recall

**Question.** The train/val/test split is homology-separated at the **whole-protein**
level (GraphPart `needle --threshold 0.3`). Does that also make the peptide
*segments* novel, or do conserved peptide motifs leak across the split? And does
peptide novelty explain where the model fails?

**Method.** Extract peptide / propeptide segments (merged coordinates) per split,
dedupe to unique sequences, and align every unique held-out (valid/test) peptide
against all train peptides of the **same type** with EMBOSS `needleall` (global
Needleman–Wunsch — the aligner family GraphPart uses). For each held-out peptide
keep its **maximum identity to any train peptide** (identity = matches /
alignment-length, GraphPart-consistent). Coverage = matches / min(len) is also
recorded to expose short-in-long containment that alignment-length identity
under-counts. Reproduce: `analysis/peptide_similarity.py` →
`analysis/peptide_similarity/peptide_similarity.csv`.

## 1. Held-out peptides are genuinely novel

![Peptide novelty histogram](figures/peptide_similarity_hist.png)

| split / type | n unique | median max-identity | ≥70% identity | ≥70% coverage |
|---|---:|---:|---:|---:|
| test / pep | 852 | 0.37 | **6%** | 8% |
| test / propep | 1118 | 0.35 | **0%** | 2% |
| valid / pep | 747 | 0.34 | 5% | 9% |
| valid / propep | 1068 | 0.36 | 1% | 4% |

Only ~6% of test peptides (and essentially 0% of propeptides) have a ≥70%-identical
counterpart in train; the median peptide sits at ~0.35 identity, which is close to
the global-alignment baseline for unrelated short sequences. So the protein-level
30% split **does** carry down to the peptide level: the held-out evaluation measures
generalization to **unseen** peptides, not motif memorization. (Caveat: using
*coverage* instead of alignment-length identity only lifts the "similar" fraction to
8–9% — a handful of test peptides are short fragments contained in a longer train
peptide; it does not change the picture.)

## 2. Novelty tracks recall — coverage, not architecture, is the ceiling

Per organism (test peptides, organisms with ≥20 segments), mean peptide
similarity-to-train vs the model's peptide recall:

![Recall vs similarity by organism](figures/recall_vs_similarity_organism.png)

| organism | mean identity to train | peptide recall | n |
|---|---:|---:|---:|
| Cyriopagopus hainanus | 0.31 | **0.05** | 495 |
| Bos taurus | 0.37 | 0.35 | 204 |
| Rattus norvegicus | 0.37 | 0.47 | 243 |
| Homo sapiens | 0.37 | 0.38 | 270 |
| Mus musculus | 0.38 | 0.53 | 207 |
| Bombyx mori | 0.38 | 0.90 | 549 |
| Caenorhabditis elegans | 0.46 | 0.43 | 495 |
| Agrotis ipsilon | 0.61 | 0.76 | 216 |

Correlation r = **0.56**. The single worst organism — *Cyriopagopus hainanus*
(spider venom), recall 0.05 — is also the **least similar to train** (0.31, i.e. no
meaningfully similar train peptide exists), and the most similar organism
(*Agrotis*, 0.61) is recovered well (0.76). This is the clean form of the
data-ceiling argument: the model fails precisely where the training data does not
cover the peptide space.

The correlation is moderate, not perfect, because **train abundance** is a second
axis: *Bombyx mori* has only middling similarity (0.38) yet 0.90 recall — it is
heavily represented in train by count (549 test segments, and correspondingly many
in train), so the model learns its peptide grammar despite low pairwise identity.
So recall is governed by *coverage of the peptide space* (both similarity AND
abundance in train), not by architecture — consistent with the near-flat
differences across the architecture/embedding tables.

## 3. Artifact for downstream use

`analysis/peptide_similarity/peptide_similarity.csv` (one row per unique held-out
peptide: seq, type, split, length, n_occurrences, organisms,
`max_identity_to_train`, `coverage_at_best`, `best_train_seq`, `is_similar_70`)
is the join key for the planned **AHO analysis** — "how much does the AHO prior help
on peptides similar to train (≥70%) vs novel ones" — and for any per-peptide
similarity-stratified error breakdown.
