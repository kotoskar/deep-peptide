# Dataset Statistics: uniprot_2022

## Summary

The uniprot_2022 dataset contains **8,449 proteins** in `labeled_sequences.csv`, of which **7,623 have a graphpart assignment** (826 excluded from all splits). Across all assigned proteins there are **5,180 peptide segments** and **7,328 propeptide segments** (counted as runs of '1' in the mask). The median peptide segment length is **21.0 aa**; **142 peptide segments** have length ≤ 5 (all exactly 5 — confirmed at both mask-run and coordinate entry level; 5 aa appears to be the true annotation floor, not a mask-merging artifact). The data is NOT concentrated on a single organism: the most common species is *Homo sapiens* (429 proteins, 5.6% of assigned). The dataset is multi-species with strong venom-organism representation (spiders, cone snails) alongside standard model organisms. GraphPart homology partitioning may place entire genera almost entirely in one split.

> **NOTE (organism format):** 348 organism strings have no parenthesis. In uniprot_2022, parentheses hold common names (e.g. 'Homo sapiens (Human)'). In uniprot_2026, most entries are plain scientific names with no parenthetical, or use parentheses for strain info. The `parse_species` function handles both correctly (takes the part before the first '(').

> **NOTE (no negatives):** This dataset contains 0 negative proteins (all assigned proteins have ≥1 peptide or propeptide segment). Negatives exist at the **residue** level (the majority of residues in positive proteins are labeled '0'), but there are no all-negative protein entries. This likely reflects that `labeled_sequences.csv` was built by selecting only UniProt proteins with at least one Chain/Peptide/Propeptide annotation.

> **NOTE (overlapping peptides):** Coordinate entries vs mask-run segments differ — peptides: 6,169 coord entries vs 5,180 mask runs; propeptides: 7,359 coord entries vs 7,328 mask runs. Overlapping/nested annotations collapse into a single mask run.


---

## 1. Data Integrity Checks

| Check | Value |
|-------|-------|
| Total proteins in labeled_sequences | 8,449 |
| Proteins with graphpart assignment | 7,623 |
| Proteins excluded (no graphpart) | 826 |
| Duplicate protein_id in labeled_sequences | 0 |
| Duplicate AC in graphpart | 0 |
| is_peptide mask length mismatches | 0 |
| is_propeptide mask length mismatches | 0 |
| Null/NaN is_peptide masks | 0 |
| Null/NaN is_propeptide masks | 0 |
| Organism strings without parenthesis | 348 |

## 2. Protein-Level Counts per Split

| Split | Total | Has Peptide | Has Propeptide | Has Both | Negatives |
|-------|-------|-------------|----------------|----------|-----------|
| TRAIN | 4,455 | 1,938 | 3,660 | 1,143 | 0 |
| VALID | 1,630 | 851 | 1,432 | 653 | 0 |
| TEST | 1,538 | 723 | 1,300 | 485 | 0 |
| ALL | 7,623 | 3,512 | 6,392 | 2,281 | 0 |

## 3. Segment-Level Counts per Split

Note: segments counted as runs of '1' in the mask (overlapping annotations collapse into one run).

| Split | Peptide Segs (mask) | Coord Entries (pep) | Propeptide Segs (mask) | Coord Entries (pro) |
|-------|---------------------|---------------------|------------------------|---------------------|
| TRAIN | 3,105 | 3,761 | 4,353 | 4,377 |
| VALID | 996 | 1,203 | 1,560 | 1,561 |
| TEST | 1,079 | 1,205 | 1,415 | 1,421 |
| ALL | 5,180 | 6,169 | 7,328 | 7,359 |

### Segments per Positive Protein

| Split | Peptide segs/protein (pos only) | Propeptide segs/protein (pos only) |
|-------|---------------------------------|------------------------------------|
| TRAIN | mean=1.60 median=1.0 | mean=1.19 median=1.0 |
| VALID | mean=1.17 median=1.0 | mean=1.09 median=1.0 |
| TEST | mean=1.49 median=1.0 | mean=1.09 median=1.0 |
| ALL | mean=1.47 median=1.0 | mean=1.15 median=1.0 |

## 4. Length Distributions

### 4a. Protein Sequence Lengths

| Split | n | min | median | mean | p90 | max |
|-------|---|-----|--------|------|-----|-----|
| TRAIN | 4,455 | 8 | 173.0 | 268.9 | 542.0 | 3971 |
| VALID | 1,630 | 8 | 91.0 | 174.1 | 437.4 | 1744 |
| TEST | 1,538 | 8 | 113.0 | 237.6 | 531.9 | 3625 |
| ALL | 7,623 | 8 | 138.0 | 242.3 | 517.8 | 3971 |

### 4b. Peptide Segment Lengths

| Split | n segs | min | median | mean | p90 | max |
|-------|--------|-----|--------|------|-----|-----|
| TRAIN | 3,105 | 5 | 19.0 | 21.7 | 39.0 | 336 |
| VALID | 996 | 5 | 24.0 | 23.6 | 36.0 | 137 |
| TEST | 1,079 | 5 | 26.0 | 25.5 | 40.2 | 71 |
| ALL | 5,180 | 5 | 21.0 | 22.9 | 39.0 | 336 |

### 4c. Propeptide Segment Lengths

| Split | n segs | min | median | mean | p90 | max |
|-------|--------|-----|--------|------|-----|-----|
| TRAIN | 4,353 | 5 | 22.0 | 22.1 | 40.0 | 50 |
| VALID | 1,560 | 5 | 22.0 | 21.1 | 31.0 | 56 |
| TEST | 1,415 | 5 | 23.0 | 22.1 | 35.0 | 79 |
| ALL | 7,328 | 5 | 22.0 | 21.9 | 37.0 | 79 |

### 4d. Tiny Peptide Segments (ALL split)

Peptide segments with length ≤ 5: **142** (all exactly 5 aa — lengths 1–4 are absent).

The minimum of 5 aa is confirmed at both the mask-run level and the coordinate-entry level: zero coordinate entries have end−start+1 < 5 in either dataset. This is a true annotation floor, NOT an artifact of overlapping peptides merging short segments into longer runs. Implication for downstream 'exclude tiny peptides' decisions: a cutoff of < 5 would remove nothing; a cutoff of ≤ 5 would remove exactly the 5-aa segments listed above.

Counts for individual lengths 1–10 (ALL split):

| Length | Count |
|--------|-------|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 142 |
| 6 | 118 |
| 7 | 185 |
| 8 | 161 |
| 9 | 208 |
| 10 | 245 |

## 5. Per-Organism Distribution (Top 15 Species)

| Species | TRAIN | VALID | TEST | TOTAL |
|---------|------ | ------ | ------ | ------|
| Homo sapiens | 291 | 68 | 70 | 429 |
| Mus musculus | 295 | 53 | 65 | 413 |
| Rattus norvegicus | 184 | 43 | 49 | 276 |
| Arabidopsis thaliana | 138 | 58 | 69 | 265 |
| Cyriopagopus hainanus | 62 | 115 | 81 | 258 |
| Bos taurus | 144 | 25 | 31 | 200 |
| Lycosa singoriensis | 1 | 125 | 40 | 166 |
| Saccharomyces cerevisiae | 127 | 17 | 20 | 164 |
| Candida albicans | 77 | 16 | 18 | 111 |
| Sus scrofa | 74 | 13 | 15 | 102 |
| Gallus gallus | 63 | 9 | 7 | 79 |
| Chilobrachys guangxiensis | 11 | 63 | 1 | 75 |
| Drosophila melanogaster | 32 | 16 | 20 | 68 |
| Conus textile | 43 | 3 | 21 | 67 |
| Canis lupus familiaris | 40 | 9 | 11 | 60 |

*Homo sapiens* accounts for **429 / 7,623 = 5.6%** of assigned proteins. The dataset is **not human-dominated**: venom organisms (spiders, cone snails) and other eukaryotes each contribute comparable numbers. GraphPart homology-based clustering may place entire genera almost exclusively in one split (e.g. *Cyriopagopus hainanus* and *Lycosa singoriensis* in uniprot_2026 are almost entirely in TRAIN), which drives per-split differences in protein length distributions and class imbalance ratios.


## 6. Residue-Level Class Imbalance

| Split | Total Residues | Peptide Residues | % Peptide | Propeptide Residues | % Propeptide |
|-------|---------------|-----------------|-----------|---------------------|--------------|
| TRAIN | 1,197,759 | 67,365 | 5.62% | 96,019 | 8.02% |
| VALID | 283,722 | 23,476 | 8.27% | 32,978 | 11.62% |
| TEST | 365,409 | 27,526 | 7.53% | 31,281 | 8.56% |
| ALL | 1,846,890 | 118,367 | 6.41% | 160,278 | 8.68% |


## 7. Histograms

See `analysis/plots/` for histogram PNGs.
