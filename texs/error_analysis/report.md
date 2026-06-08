# Error analysis: where the peptide predictor fails

**Scope.** Inference on the held-out TEST partition (GraphPart cluster 4) for the
top-5 models of each results table (architectural changes ∪ embedding generators),
9 runs total, 29 514 true/predicted segments. Per-segment outcomes are aggregated
by **peptide length** and by **organism** to locate the systematic failure modes.

**Matching rule.** A true peptide (or propeptide) group counts as *recovered* iff
some prediction has BOTH its start and its stop within **±3 residues** of a group
member — the acceptance window used throughout DeepPeptide. Overlapping true
annotations are collapsed into one group (recovering any member is enough).
Recall = recovered / total true groups; a prediction matching no true group is a
false positive.

> All recall numbers here use a **corrected** implementation of the ±3 matcher.
> The shipped metric (`manuscript_metrics.get_counts_for_protein`) has a bug that
> understates recall by ~2–4 pp; see the last section. It does not change *relative*
> conclusions (the bug is a near-constant offset across runs and length/organism bins).

Reproduce: `env/bin/python analysis/error_analysis.py` → `analysis/error_stats/`;
figures via `env/bin/python analysis/plot_error_analysis.py`.

---

## 1. Length is the dominant factor, and it is not the short peptides

![Recall by peptide length](figures/recall_by_length.png)

| length (aa) | peptides recall | propeptides recall |
|---|---:|---:|
| 5 | 0.36 | 0.35 |
| 6–10 | **0.75** | 0.29 |
| 11–20 | 0.69 | 0.57 |
| 21–30 | 0.60 | **0.79** |
| 31–50 | 0.38 | 0.48 |

Peptides and propeptides have **opposite** length profiles. Peptides are recovered
best in the 6–20 aa range and degrade sharply for **long** segments (31–50 aa →
0.38). Propeptides are worst for **short** segments (6–10 aa → 0.29) and best at
21–30 aa. True segments are capped at 50 aa in this dataset/label encoding (the
multistate CRF has 50 length states per class), so there is no 51+ bin.

Where the actual error mass sits (count of missed segments, not rate):

![FN mass by length](figures/fn_mass_by_length.png)

For **peptides** the misses concentrate in the 21–30 and 31–50 bins (~1660 FN
each) — i.e. the model loses the most ground on long peptides, both because recall
is low there and because those bins are well populated.

### Tiny peptides (length 5) are *not* the problem

A recurring question is whether very short peptides (≤5 aa, comparable to the ±3
window itself) should be excluded. They should not, on error grounds:

- **Peptides:** 162 true len-5 segments, recall 0.36 — but only **104 of 4359 total
  FN (2.4%)**.
- **Propeptides:** 252 true len-5 segments, recall 0.35 — only **164 of 5339 FN (3.1%)**.

Excluding len-5 segments removes <3 % of the errors and a few hundred examples.
The leverage is on long peptides and on under-represented organisms (below), not
on tiny ones.

---

## 2. Organism is the second axis: under-representation in train drives failure

![Recall by organism](figures/recall_by_organism.png)

Peptide recall varies from **0.05 to 0.90** across the most frequent organisms:

| organism | recall | group |
|---|---:|---|
| Bombyx mori | 0.90 | good (>0.7) |
| Procambarus clarkii | 0.90 | good (>0.7) |
| Conus textile | 0.79 | good (>0.7) |
| Agrotis ipsilon | 0.76 | good (>0.7) |
| Drosophila melanogaster | 0.75 | good (>0.7) |
| Homo sapiens | 0.38 | poor (<0.4) |
| Bos taurus | 0.35 | poor (<0.4) |
| Aplysia californica | 0.27 | poor (<0.4) |
| Cyriopagopus hainanus | 0.05 | poor (<0.4) |

The standout failure is **Cyriopagopus hainanus** (spider venom, 495 test peptides,
recall 0.05 — essentially never recovered). This tracks the GraphPart split: venom
organisms like *Cyriopagopus* and *Lycosa* are concentrated in the valid/test
partitions and barely present in train (see `analysis/dataset_stats_2022.md`), so
the model never learns their peptide grammar. **Homo sapiens recall is only 0.38**,
which is relevant because the "homo-only" evaluation slice is both small and on the
hard side of the distribution.

Takeaway: the ceiling here is set by **train coverage of organism/peptide-family
diversity**, not by model architecture — consistent with the near-flat metric
differences across the architecture and embedding tables.

---

## 3. Per-run view

![Per-run recall](figures/recall_by_run.png)

Recall differences between the top architectures/embeddings are small (peptide
recall ~0.50–0.61) relative to the length- and organism-driven spread above. No
single top model rescues the hard bins; the failure structure is shared.

---

## 4. The shipped metric understates recall (bug — inherited from upstream DeepPeptide)

`manuscript_metrics.get_counts_for_protein` (the ±3 peptide-finding metric behind
every reported P/R/F1) has a **variable-shadowing bug**:

```python
for idx, row in true_df.iterrows():        # idx = true-row index
    ...
    for idx, row in pred_df.iterrows():    # rebinds idx to the PRED index
        if start_match and stop_match:
            true_df.loc[idx, 'matched'] = True   # writes to the PRED-labelled row, not the true row
            break
```

When a true peptide matches, the `matched` flag is written to `true_df.loc[<pred
index>]` instead of the current true row. If a protein has **more predictions than
true segments**, the matching pred's index exceeds the true index range, so
`.loc` *creates a phantom row* with `group = NaN`, which `groupby('group')` then
silently drops — the match is lost. A perfectly predicted peptide can therefore be
counted as a miss:

```
true=[(10,30)]  pred=[(200,220),(10,30)]   # 2nd prediction is an exact hit
  shipped metric (tp,fn,fp) = (0, 1, 1)    # counted as a MISS
  corrected      (tp,fn,fp) = (1, 0, 1)
```

**This bug is present in the upstream DeepPeptide code too** (verified), so the
numbers reported here remain directly comparable to the original paper — which is
exactly why we *keep* the published values and report a corrected column alongside
them rather than overwriting. Empirically the corrected ±3 recall is a near-constant
**+0.024…+0.044 (peptides) / +0.011…+0.023 (propeptides)** above the published value
across all runs, so model rankings are unchanged. The full original-vs-corrected
table is in `analysis/corrected_metrics.csv` / the canonical metrics report.

*Fix (if ever desired):* give the inner loop its own variable
(`for p_idx, p_row in pred_df.iterrows(): ... true_df.loc[t_idx,'matched']=True;
pred_df.loc[p_idx,'matched']=True`). Applying it would shift every reported number
up by ~2–4 pp and break comparability with the paper, hence left as a documented,
dual-reported finding.
