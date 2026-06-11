# Why ESM-C 6B + boundary head misses propeptides at the length extremes (6-10 and 31-50)

Data: `analysis/errors/error_stats/esmc6b_boundary_bond__segments.csv` (1415 true test-set
propeptides, joined by protein_id+length to `data/uniprot_2022/labeled_sequences.csv` for
coordinates/sequence — 1415/1415 rows joined cleanly).

Length-bin overview (matches the reported U-shape, confirmed against `task_confusion.csv`):

| length_bin | n   | missed_rate (matched=False) | correct | confused_as_other |
|---|---|---|---|---|
| 6-10  | 317 | **0.697** | 0.303 | 0.123 |
| 11-20 | 266 | 0.432 | 0.566 | 0.139 |
| 21-30 | 545 | **0.224** | 0.776 | 0.081 |
| 31-50 | 287 | **0.502** | 0.498 | 0.223 |

(`matched=False` includes "confused" so it's slightly higher than the pure "missed" rates
quoted in the task description, but the shape is identical: best at 21-30, much worse at
both tails, and 31-50 has visibly more cross-type confusion than 6-10.)

## Top findings (number-backed)

### 1. The 21-30 bin is dominated by C-terminal propeptides; this is where the model excels — and it is *not* an organism artefact

For 21-30, 51.7% of propeptides have `end/seqlen > 0.9` (i.e. sit in the last 10% of the
sequence — classic C-terminal "extension peptide" position), and missed-rate among those
is only **0.191** vs 0.259 for the rest. C-terminal propeptides preceded by the mature
peptide are the model's best-understood case.

Within single organisms the U-shape persists (so it is *not* purely the organism-recall
confound):

| organism (n propeptides) | 6-10 | 11-20 | 21-30 | 31-50 |
|---|---|---|---|---|
| Homo sapiens (n=59) | 0.857 (n=7) | 0.400 (n=10) | 0.241 (n=29) | 0.692 (n=13) |
| Bos taurus (n=27)   | 0.800 (n=5) | 0.375 (n=8)  | 0.250 (n=8)  | 0.833 (n=6)  |

Both show the same dip-then-rise pattern as the global numbers, despite very different
overall organism recall (Arabidopsis missed=0.156, Bombyx mori missed=0.706, Bos taurus
0.519). So while organism is a real confound for the *overall level*, the *length shape*
is a separate, additional effect present within organisms too.

### 2. 6-10 propeptides: structurally different positional profile, dominated by proteins with NO labeled mature peptide at all, and an enrichment of P/R/G/Q/T (low-complexity / linker-like) over L/I/F/V (hydrophobic core-like)

- **Adjacency**: 256/317 (81%) of 6-10 propeptides are in proteins with **zero** labeled
  peptide segments (`n_peptides==0`); only 61/317 (19%) sit within ±3 residues of a
  peptide. Missed rate is high in *both* groups but especially for the no-peptide-anchor
  group: 0.707 (no peptide in protein) vs 0.656 (adjacent to a peptide). For comparison,
  in 21-30 the no-peptide group has missed rate 0.184 — almost identical to the
  peptide-adjacent group (0.221). So lacking a peptide "anchor" hurts much more for short
  propeptides.

- **Position is bimodal and the two modes behave very differently**: among *matched*
  (correctly detected) 6-10 propeptides, 74% (71/96) sit in the first 30% of the sequence
  (classic signal/transit-peptide-like N-terminal slot) and another 23% sit in the last
  10% (C-terminal). Among *missed* 6-10 propeptides, only 29% are in the first 30%, while
  44% (98/221) are in the last 10% (C-terminal) and a further 21% sit in the *middle*
  (rel. position 0.3-0.5) — a slot essentially absent (1/96) among matched ones. So the
  model's "short propeptide" template is effectively "near the very start of the chain";
  short propeptides located mid-chain or at the very end are systematically missed.

- **AA composition**: comparing all missed vs matched propeptides (any length), missed
  ones are enriched in P (+0.018), R (+0.016), G (+0.015), Q (+0.012), T (+0.010) and
  depleted in L (-0.034), I (-0.009), F (-0.009), D (-0.009), V (-0.007) — i.e. missed
  segments skew toward Pro/Gly/Arg/Gln-rich, lower-complexity, less-hydrophobic stretches,
  while matched ones are more Leu/hydrophobic. This holds specifically for the 6-10 bin:
  missed 6-10 propeptides' top residues are A, E, G, N, R, S (no L in top 6), whereas
  21-30 and 31-50 missed sets both have L as the #1 residue. Hydrophobic fraction for
  6-10 is 0.322 (matched) vs 0.288 (missed) — the largest matched-vs-missed gap of any
  length bin (other bins differ by ≤0.05). This is consistent with short, hydrophobic,
  N-terminal propeptides resembling signal/transit peptides (a class the model is
  presumably good at via ESM-C's pretraining), while non-hydrophobic, mid/C-terminal
  short propeptides have no such recognizable template.

### 3. 31-50 propeptides: the telescoping CRF length cap (50) bites hard, and confusion (not just misses) rises

- The model's max representable segment length is 50. Missed rate for length 45-50 is
  **0.786** (n=42) vs **0.453** for length 31-44 (n=245) — a ~1.7x jump right at the cap.
  Going length-by-length within 31-50, missed rate is mostly <0.4 for L=31-39 except for
  noisy spikes (33,34,36 ~0.6), then climbs to 0.6-0.89 for L=42-50, with 47/48/50 at
  0.83-0.89. The maximum observed true propeptide length in the test set is exactly 50
  (n=6, missed 0.833) — i.e. there is no length >50 to check, but the approach to 50 is
  visibly where things degrade.
- Unlike 6-10 (mostly pure "missed"), the 31-50 bin shows elevated `confused_as_other`
  (0.223 vs 0.123 for 6-10 and 0.081 for 21-30) — long propeptides that the model detects
  *as a segment* but mislabels as "peptide" rather than "propeptide", on top of the
  length-cap-driven pure misses.
- 31-50 propeptides also have the *highest* mean n_propeptides-per-protein (1.39) and
  highest mean n_peptides-per-protein (1.29) of any bin — they tend to occur in
  multi-segment precursors (e.g. spider/scorpion toxin precursors with several
  propeptide+peptide repeats), which independently raises difficulty (see point 4).

### 4. More propeptides per protein -> much higher miss rate, regardless of length

Missed rate by `n_propeptides` in the protein: 1 propeptide -> 0.382, 2 -> 0.713,
3 -> 0.615, 4 -> 0.857 (n=35), 5 -> 0.400 (small n), 7 -> 0.429 (small n). Proteins with
≥2 propeptide segments are missed roughly 1.6-2.2x more often than single-propeptide
proteins. The 31-50 bin is enriched for these multi-segment precursors (mean
n_propeptides=1.39 vs 1.15-1.36 for the other bins), which is part of why it
underperforms relative to 21-30 even though its lengths are not all near the cap.

## What did NOT show a strong effect

- **Cysteine content**: no consistent matched-vs-missed pattern (e.g. 21-30:
  matched 0.0092 vs missed 0.0058; 31-50: matched 0.0048 vs missed 0.0110 — opposite
  signs across bins, not a clean signal).
- **Sequence length of the host protein**: mean seqlen per bin (185, 219, 288, 221 for
  6-10/11-20/21-30/31-50) doesn't track the missed-rate U-shape in an obvious
  monotonic way; 21-30 (best detected) sits in the *longest* proteins on average, so
  "short proteins are easier" is not the explanation.
- **"Isolated" propeptides** (peptide present in protein but >3 residues away, no
  overlap/adjacency) are rare (37/1415, mostly in 21-30 and 31-50) and have a notably
  high missed rate (0.757 overall, 0.692 in 21-30, 1.0 in 31-50/n=9) — but n is too small
  to be a primary driver of the bin-level effect; it's a secondary contributor.

## Summary / biological framing

1. **6-10 (worst, 70% missed):** the model has effectively learned "short propeptide ==
   N-terminal, hydrophobic, signal/transit-peptide-like region anchored next to (or just
   before) a mature peptide." Short propeptides that violate this template — sitting
   mid-chain or C-terminal, in proteins with no annotated mature peptide, with
   Pro/Gly/Arg/Gln-rich rather than hydrophobic composition — are the ones missed. These
   look like a structurally/biologically distinct subclass (possibly linker or
   spacer-type short propeptides rather than signal-peptide-like ones) that the training
   distribution under-represents in non-canonical positions.

2. **21-30 (best, 22% missed):** dominated by C-terminal "extension peptide" propeptides
   immediately following a mature peptide (52% are in the last 10% of the sequence) — the
   canonical, well-represented precursor architecture (e.g. neuropeptide precursors).

3. **31-50 (50% missed, more confusion):** two compounding effects — (a) approaching the
   CRF's hard length cap of 50 residues (missed rate jumps from 0.45 to 0.79 for L≥45),
   and (b) these segments occur disproportionately in multi-propeptide/multi-peptide
   precursors (mean 1.39 propeptides/protein, the highest of any bin), where having
   multiple segments to place independently roughly doubles the per-segment miss rate
   (0.38 for single-propeptide proteins vs 0.61-0.86 for proteins with ≥2-4).

All three explanations are quantitatively grounded in the joined segment/coordinate data
above; no GPU inference was used.
