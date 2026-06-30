# Report & presentation figure generators

All scripts are run **from the repo root** with the project env, e.g.:

```bash
env/bin/python analysis/metrics/src/generators/ladder_fig.py
```

Paths inside are repo-root-relative. GPU scripts set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for determinism; figure scripts are CPU-only
(they read the CSVs the GPU scripts produce). Figures are written to both
`analysis/metrics/figures/` and `texs/Overleaf/figures/`.

## Presentation pass (`_pres.py` + `PRES=1`)
`_pres.py` holds the **canonical** `MODEL_LABELS` (full modification chain, e.g.
`ESM-C 6B + boundary + gated(256) + 3Di`) and `COLORS`, imported by every
multi-model generator so legends/colours are unified everywhere and the baseline
reads `ESM-2 (бейзлайн)`. Run any report generator with `PRES=1` to emit a
**clean presentation copy** into `presentation/figures/` only: titles drop the
parenthetical/`(а)(б)`/fold-union chrome (`clean_title`) and gray helper text is
suppressed. Without `PRES`, generators keep their original report output. Example:

```bash
PRES=1 env/bin/python analysis/metrics/src/generators/report_fig_tolerance.py
```

## Figure generators (CPU, read CSVs)
| script | produces |
|---|---|
| `report_figs_data.py` | `data_distributions.png`, `fold_divergence.png` (per-fold length profiles) |
| `report_figs_v2.py` | `scoreboard.png` (also writes a 1-panel `trades.png` — run `trades_fig.py` AFTER) |
| `trades_fig.py` | `trades.png` (2-panel ΔP/ΔR: 3Di + bond) — **run after** `report_figs_v2.py` |
| `trades_3di_fig.py` | `trades_3di.png` (single panel, 3Di trade only, no bond) |
| `effects_slopes_fig.py` | `effects_slopes.png` — 3Di + bond as boundary-style **slope** panels (F1 before→after, 2 lines pep/propep, models named below); presentation replacement for the trades histograms |
| `length_profile_f1_fig.py` | `length_profile_f1.png` — 2-panel: per-fold length profile + F1-by-length (5 models, no P/R) |
| `report_fig_tolerance.py` | `tolerance.png` |
| `report_fig_interaction.py` | `interaction.png` |
| `bylength_fig.py` | `bylength.png` (5 ladder models incl. the gated adapter) |
| `datascale_plot.py` | `datascale_curve.png` |
| `datascale_tol_plot.py` | `datascale_tolerance.png` |
| `sim_fig_report.py` | `similarity.png` |
| `ladder_fig.py` | `figures/ladder/` — `ladder.png`, `ladder_tol.png` (tolerance panel spans ±5..±0, retention normalized to ±3) + build-up frames `f1..f6`. Reads `ladder_tol_ext.csv`. |

## Data generators (GPU inference → CSVs)
| script | produces |
|---|---|
| `thesis_eval.py` | `clean_tol_true.csv`, `clean_tol_pred.csv` (per-segment ±tol; foundational) |
| `corrected_scoreboard.py` | `clean_split_modelselect.csv` (corrected vs bug matcher, fold 2) |
| `gpu_fig_data.py` | `interaction_perprotein_2026.csv`, `similarity/seg_matched_zeroctrl.csv` |
| `sim_infer.py` | `similarity/seg_matched_2026.csv` |
| `sim_identity.py` | `similarity/identity_2026.csv` (needleall segment identity to train) |
| `datascale_curve.py` | `datascale_curve.csv` |
| `datascale_tol2.py` | `datascale_tol_perprotein.csv` |
| `ladder_tol_ext_eval.py` | `ladder_tol_ext.csv` (per-protein tp/fn/fp at tols 0..5 for the 6 ladder-tol models; feeds `ladder_fig.py`) |
| `effects_esm2_eval.py` | `effects_esm2_extra.csv` (per-segment task-split tp/fn/fp at ±3 for `esm2_boundary` / `esm2_3di_proj_gated_conv`, missing from `clean_tol_*`) |

## Gated-adapter ablation pipeline (isolates the gated seq-adapter)
| script | role |
|---|---|
| `train_adapter_ablation.py` | trains `runs/2026_esmc6b_adapterNNN_seqonly` = zeroctrl recipe + `--gated_seq_only` (seq adapter, no struct). Usage: `<seq_proj> <out_dir> [epochs]` |
| `eval_adapter256.py` | `adapter256_perprotein_2026.csv` + paired adapter Δ vs orange |
| `eval_adapter256_tol.py` | `adapter256_tol_perprotein.csv` (tolerance curve) |
| `eval_adapter256_seg.py` | `adapter256_seg_true.csv` + `adapter256_seg_pred.csv` (per-segment lengths, for bylength) |
| `esm2_boundary_infer.py` | ESM-2 boundary-head Δ (baseline_esm2 vs esm2_boundary) |
