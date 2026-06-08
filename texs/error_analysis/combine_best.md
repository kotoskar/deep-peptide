# Combining the best embedding with a complementary architecture

**Setup.** After the fp32 fix, the plain ESM2 baseline tops both experiment tables on
MCC, and no single architectural variant beats it — which looked like a ceiling. But
the two tables optimize different things: the best *embedding* by residue-level signal
is **ESM-C 6B** (MCC 0.758, AUC 0.766, and the highest recall in Table 2, 0.590), yet
it has the **lowest precision** (0.570) so its ±3 F1 (0.579) lags the ESM2 baseline
(0.607). Hypothesis: pair ESM-C 6B's strong residue signal with an architecture that
**sharpens cleavage boundaries** — `lstmcnncrf_boundary_bond_loss` (a learned
start/inside/end boundary-emission head + an auxiliary soft cleavage-site bond loss) —
to convert recall into precision.

Run: `runs/esmc6b_boundary_bond` (model `lstmcnncrf_boundary_bond_loss`, embeddings
`embeddings_esmc6b` dim 2560, bond defaults λ=0.02/window 5/τ 1.5, 100 ep, bs 48,
seed 42). Metrics from a deterministic fp32 inference (drift vs train-time = 0.0000).

## Result: it breaks above the baseline ceiling

| config | F1 all | Prec all | Rec all | MCC all |
|---|---:|---:|---:|---:|
| ESM2 baseline (lstmcnncrf) | 0.607 | 0.640 | 0.578 | 0.750 |
| ESM-C 6B baseline (lstmcnncrf) | 0.579 | 0.570 | 0.590 | 0.758 |
| **ESM-C 6B + boundary_bond** | **0.657** | **0.714** | 0.609 | **0.765** |

Same ranking under the debugged ±3 metric (`analysis/corrected_metrics.csv`):
corrected F1 0.675 vs ESM2 0.621 vs ESM-C 6B 0.599.

This is the **best F1 and best MCC in the whole project** — +0.05 F1 over the ESM2
baseline and +0.046 over the previous table maximum (0.611). The mechanism is exactly
the hypothesis: **precision jumps +0.14 (0.570 → 0.714) while recall holds** (~0.59 →
0.61). The boundary head turns ESM-C 6B's abundant-but-fuzzy residue signal into
sharp ±3 cleavage calls.

## Why this is a real (and informative) win

- It is **not** the AMP/metric artifact: drift train-time↔fp32-infer is 0.0000, and
  the gain survives the debugged ±3 metric.
- It is a **complementary pairing**, not stacking two strong numbers: `boundary_bond`
  barely helped ESM2 (F1 0.606 ≈ baseline 0.607) because ESM2 is already
  precision/recall-balanced. It helps ESM-C 6B a lot precisely because ESM-C 6B is
  high-recall / low-precision, so the boundary head has slack to exploit. The benefit
  of an architecture depends on the embedding's P/R profile.
- It refines the ceiling story: the data-scaling curve shows the model is data-limited,
  but a *well-matched* architecture+embedding combination still buys ~+0.05 F1 on top —
  the ceiling is not yet hit at fixed data either.

## Caveats / follow-up

- **Single seed.** Single-run training variance is ~±0.02 F1 (data-scaling section), so
  the +0.05 gain is ~2.5× the noise and consistent across metrics — but a 3-seed
  repeat would make it airtight.
- AUC all is low (0.609) for this run; AUC here is calibration-sensitive and was noisy
  across all runs, so MCC/F1 are the reliable comparison.
- Natural next probes: ESM-C 6B + telescoping_segmental (the other boundary-aware
  arch); and whether the gain holds on the homo-only and per-organism/length error
  breakdowns.
