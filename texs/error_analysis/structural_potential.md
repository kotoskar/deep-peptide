# Do structural embeddings have the same latent potential as ESM-C 6B?

The P2a win (ESM-C 6B + boundary/bond) worked through a specific mechanism: ESM-C 6B
is **high-recall / low-precision**, and the boundary head converted recall into
precision (+0.14). Natural question: do the **structural** embeddings (AFT, 3Di,
ProstT5) hide a similar exploitable profile, and should they be combined with — or
replace — ESM-C 6B?

## 1. Structural embeddings have the OPPOSITE profile — the boundary trick won't transfer

| embedding (TEST) | F1 | Prec | Rec | profile |
|---|---:|---:|---:|---|
| **ESM-C 6B** | 0.579 | 0.570 | 0.590 | **high-recall / low-precision** ✅ |
| ESM2 baseline | 0.607 | 0.640 | 0.578 | balanced |
| ESM2+3Di proj.gated.conv | 0.611 | 0.658 | 0.571 | precision-heavy |
| ESM2+AFT (pair, gated) | 0.595 | 0.651 | 0.547 | precision-heavy |
| ProstT5 | 0.509 | 0.588 | 0.449 | precision-heavy, recall-poor |
| AFT (no lddt) | 0.408 | 0.458 | 0.368 | weak both |
| AFT | 0.382 | 0.529 | 0.299 | precision-heavy, recall-starved |

Only ESM-C 6B has recall > precision. Every structural config is precision-heavy and
**recall-starved** — they miss peptides rather than mis-place boundaries. So a
boundary-sharpening head (which trades recall→precision) is the wrong tool for them; if
anything they need the opposite (a recall boost). **The P2a mechanism does not transfer
to structural embeddings.**

## 2. But they DO give complementary signal — in two specific places

- **Length extremes (3Di).** In the per-model recall-by-length breakdown,
  `esm2+3di_proj` is the **best model at the hardest lengths**: len 5 → 0.444 (vs
  baseline 0.389) and len 31–50 → 0.449 (vs baseline 0.405). Structural context helps
  exactly where pure-sequence models struggle most.
- **Ranking / AUC (ProstT5, AFT).** ProstT5 and AFT have high residue AUC (0.78 / 0.73)
  despite low F1 — good at *ranking* residues, poor at the ±3 threshold. That is latent
  signal a better head/threshold could exploit.
- The best embedding-table F1 was `esm2+3di_proj_gated_conv` (0.611), i.e. 3Di already
  adds a hair over ESM2 baseline.

## 3. Recommendation

- **Replace ESM-C 6B with structural? No.** Structural-only is far weaker (AFT 0.38,
  ProstT5 0.51); ESM-C 6B (a large 2560-d model) dominates. They are not substitutes.
- **Combine with ESM-C 6B? Worth one experiment.** The promising direction is to add a
  **3Di structural branch** to the ESM-C 6B + boundary winner, to pick up the
  length-extreme signal on top of the current best. This needs a new concatenated
  embedding `embeddings_esmc6b_3di` (ESM-C 6B 2560-d ⊕ ProstT5-3Di 20-d) — cheap to
  build (3Di is fast) — then train `lstmcnncrf_gated3diresidual_conv` (the
  structural-branch arch) on it, ideally with the boundary/bond loss too.
- Expectation: modest gain at the length tails; unlikely to beat the +0.05 from P2a,
  but tests whether structural and ESM-C 6B signals are additive. **Queued as a GPU
  experiment (see roadmap), to run after the telescoping candidate B finishes.**
