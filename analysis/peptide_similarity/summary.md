# Peptide-level similarity of held-out peptides to TRAIN

Per unique valid/test peptide: max identity (needle, matches/alignment_length) to any train peptide of the same type. `is_similar_70` = identity ≥ 70%. `coverage_at_best` = matches/min(len) at the best hit (catches short-in-long containment).

> The split is homology-separated at the WHOLE-PROTEIN level (GraphPart needle, 30%). This measures whether the peptide SEGMENTS are also novel.

- **valid/pep** (n=747 unique): median max-identity 0.34; **5% ≥70% identity** to a train peptide; 9% ≥70% coverage (containment).
- **valid/propep** (n=1068 unique): median max-identity 0.36; **1% ≥70% identity** to a train peptide; 4% ≥70% coverage (containment).
- **test/pep** (n=852 unique): median max-identity 0.37; **6% ≥70% identity** to a train peptide; 8% ≥70% coverage (containment).
- **test/propep** (n=1118 unique): median max-identity 0.35; **0% ≥70% identity** to a train peptide; 2% ≥70% coverage (containment).

## Max-identity-to-train distribution (test, both types)

| identity bin | pep | propep |
|---|---:|---:|
| <0.30 | 113 | 149 |
| 0.30–0.50 | 568 | 896 |
| 0.50–0.70 | 124 | 70 |
| 0.70–0.90 | 33 | 3 |
| 0.90–1.00 | 14 | 0 |
