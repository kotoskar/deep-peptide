# Dataset Statistics: uniprot_2026

## Summary

The uniprot_2026 dataset contains **9,619 proteins** in `labeled_sequences.csv`, of which **9,456 have a graphpart assignment** (163 excluded from all splits). Across all assigned proteins there are **6,137 peptide segments** and **8,960 propeptide segments** (counted as runs of '1' in the mask). The median peptide segment length is **21.0 aa**; **152 peptide segments** have length ≤ 5 (all exactly 5 — confirmed at both mask-run and coordinate entry level; 5 aa appears to be the true annotation floor, not a mask-merging artifact). The data is NOT concentrated on a single organism: the most common species is *Homo sapiens* (457 proteins, 4.8% of assigned). The dataset is multi-species with strong venom-organism representation (spiders, cone snails) alongside standard model organisms. GraphPart homology partitioning may place entire genera almost entirely in one split.

> **NOTE (organism format):** 8369 organism strings have no parenthesis. In uniprot_2022, parentheses hold common names (e.g. 'Homo sapiens (Human)'). In uniprot_2026, most entries are plain scientific names with no parenthetical, or use parentheses for strain info. The `parse_species` function handles both correctly (takes the part before the first '(').

> **NOTE (no negatives):** This dataset contains 0 negative proteins (all assigned proteins have ≥1 peptide or propeptide segment). Negatives exist at the **residue** level (the majority of residues in positive proteins are labeled '0'), but there are no all-negative protein entries. This likely reflects that `labeled_sequences.csv` was built by selecting only UniProt proteins with at least one Chain/Peptide/Propeptide annotation.

> **NOTE (overlapping peptides):** Coordinate entries vs mask-run segments differ — peptides: 7,305 coord entries vs 6,137 mask runs; propeptides: 8,992 coord entries vs 8,960 mask runs. Overlapping/nested annotations collapse into a single mask run.


---

## 1. Data Integrity Checks

| Check | Value |
|-------|-------|
| Total proteins in labeled_sequences | 9,619 |
| Proteins with graphpart assignment | 9,456 |
| Proteins excluded (no graphpart) | 163 |
| Duplicate protein_id in labeled_sequences | 0 |
| Duplicate AC in graphpart | 0 |
| is_peptide mask length mismatches | 0 |
| is_propeptide mask length mismatches | 0 |
| Null/NaN is_peptide masks | 0 |
| Null/NaN is_propeptide masks | 0 |
| Organism strings without parenthesis | 8,369 |

## 2. Protein-Level Counts per Split

| Split | Total | Has Peptide | Has Propeptide | Has Both | Negatives |
|-------|-------|-------------|----------------|----------|-----------|
| TRAIN | 5,636 | 2,940 | 4,848 | 2,152 | 0 |
| VALID | 1,912 | 488 | 1,605 | 181 | 0 |
| TEST | 1,908 | 845 | 1,411 | 348 | 0 |
| ALL | 9,456 | 4,273 | 7,864 | 2,681 | 0 |

## 3. Segment-Level Counts per Split

Note: segments counted as runs of '1' in the mask (overlapping annotations collapse into one run).

| Split | Peptide Segs (mask) | Coord Entries (pep) | Propeptide Segs (mask) | Coord Entries (pro) |
|-------|---------------------|---------------------|------------------------|---------------------|
| TRAIN | 3,801 | 4,457 | 5,413 | 5,418 |
| VALID | 915 | 1,064 | 1,813 | 1,839 |
| TEST | 1,421 | 1,784 | 1,734 | 1,735 |
| ALL | 6,137 | 7,305 | 8,960 | 8,992 |

### Segments per Positive Protein

| Split | Peptide segs/protein (pos only) | Propeptide segs/protein (pos only) |
|-------|---------------------------------|------------------------------------|
| TRAIN | mean=1.29 median=1.0 | mean=1.12 median=1.0 |
| VALID | mean=1.88 median=1.0 | mean=1.13 median=1.0 |
| TEST | mean=1.68 median=1.0 | mean=1.23 median=1.0 |
| ALL | mean=1.44 median=1.0 | mean=1.14 median=1.0 |

## 4. Length Distributions

### 4a. Protein Sequence Lengths

| Split | n | min | median | mean | p90 | max |
|-------|---|-----|--------|------|-----|-----|
| TRAIN | 5,636 | 8 | 93.0 | 186.1 | 440.0 | 3625 |
| VALID | 1,912 | 8 | 257.0 | 313.6 | 555.9 | 3971 |
| TEST | 1,908 | 8 | 150.0 | 277.2 | 550.6 | 2912 |
| ALL | 9,456 | 8 | 137.0 | 230.2 | 482.0 | 3971 |

### 4b. Peptide Segment Lengths

| Split | n segs | min | median | mean | p90 | max |
|-------|--------|-----|--------|------|-----|-----|
| TRAIN | 3,801 | 5 | 23.0 | 23.0 | 39.0 | 156 |
| VALID | 915 | 5 | 14.0 | 19.6 | 37.0 | 576 |
| TEST | 1,421 | 5 | 21.0 | 23.3 | 40.0 | 336 |
| ALL | 6,137 | 5 | 21.0 | 22.6 | 39.0 | 576 |

### 4c. Propeptide Segment Lengths

| Split | n segs | min | median | mean | p90 | max |
|-------|--------|-----|--------|------|-----|-----|
| TRAIN | 5,413 | 5 | 22.0 | 21.2 | 35.0 | 79 |
| VALID | 1,813 | 5 | 19.0 | 20.3 | 39.0 | 62 |
| TEST | 1,734 | 5 | 19.0 | 20.3 | 37.0 | 50 |
| ALL | 8,960 | 5 | 21.0 | 20.9 | 36.0 | 79 |

### 4d. Tiny Peptide Segments (ALL split)

Peptide segments with length ≤ 5: **152** (all exactly 5 aa — lengths 1–4 are absent).

The minimum of 5 aa is confirmed at both the mask-run level and the coordinate-entry level: zero coordinate entries have end−start+1 < 5 in either dataset. This is a true annotation floor, NOT an artifact of overlapping peptides merging short segments into longer runs. Implication for downstream 'exclude tiny peptides' decisions: a cutoff of < 5 would remove nothing; a cutoff of ≤ 5 would remove exactly the 5-aa segments listed above.

Counts for individual lengths 1–10 (ALL split):

| Length | Count |
|--------|-------|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 152 |
| 6 | 169 |
| 7 | 208 |
| 8 | 201 |
| 9 | 350 |
| 10 | 271 |

## 5. Per-Organism Distribution (Top 15 Species)

| Species | TRAIN | VALID | TEST | TOTAL |
|---------|------ | ------ | ------ | ------|
| Homo sapiens | 246 | 110 | 101 | 457 |
| Mus musculus | 264 | 100 | 88 | 452 |
| Rattus norvegicus | 170 | 70 | 60 | 300 |
| Arabidopsis thaliana | 131 | 91 | 62 | 284 |
| Cyriopagopus hainanus | 266 | 0 | 0 | 266 |
| Bos taurus | 114 | 52 | 47 | 213 |
| Lycosa singoriensis | 172 | 0 | 1 | 173 |
| Saccharomyces cerevisiae | 117 | 18 | 29 | 164 |
| Sus scrofa | 55 | 39 | 33 | 127 |
| Candida albicans | 40 | 44 | 29 | 113 |
| Drosophila melanogaster | 57 | 20 | 13 | 90 |
| Gallus gallus | 51 | 20 | 17 | 88 |
| Californiconus californicus | 79 | 2 | 0 | 81 |
| Chilobrachys guangxiensis | 80 | 1 | 0 | 81 |
| Conus textile | 75 | 1 | 1 | 77 |

*Homo sapiens* accounts for **457 / 9,456 = 4.8%** of assigned proteins. The dataset is **not human-dominated**: venom organisms (spiders, cone snails) and other eukaryotes each contribute comparable numbers. GraphPart homology-based clustering may place entire genera almost exclusively in one split (e.g. *Cyriopagopus hainanus* and *Lycosa singoriensis* in uniprot_2026 are almost entirely in TRAIN), which drives per-split differences in protein length distributions and class imbalance ratios.


## 6. Residue-Level Class Imbalance

| Split | Total Residues | Peptide Residues | % Peptide | Propeptide Residues | % Propeptide |
|-------|---------------|-----------------|-----------|---------------------|--------------|
| TRAIN | 1,048,708 | 87,528 | 8.35% | 114,827 | 10.95% |
| VALID | 599,599 | 17,957 | 2.99% | 36,875 | 6.15% |
| TEST | 528,820 | 33,096 | 6.26% | 35,286 | 6.67% |
| ALL | 2,177,127 | 138,581 | 6.37% | 186,988 | 8.59% |


## 7. Histograms

See `analysis/plots/` for histogram PNGs.
