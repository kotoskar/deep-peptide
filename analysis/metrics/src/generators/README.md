# Report & presentation figure generators

All scripts are run **from the repo root** with the project env, e.g.:

```bash
env/bin/python analysis/metrics/src/generators/ladder_fig.py
```

Paths inside are repo-root-relative. GPU scripts set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for determinism; figure scripts are CPU-only
(they read the CSVs the GPU scripts produce). Figures are written to both
`analysis/metrics/figures/` and `texs/Overleaf/figures/`.

## Figure generators (CPU, read CSVs)
| script | produces |
|---|---|
| `report_figs_data.py` | `data_distributions.png`, `fold_divergence.png` (per-fold length profiles) |
| `report_figs_v2.py` | `scoreboard.png` (also writes a 1-panel `trades.png` — run `trades_fig.py` AFTER) |
| `trades_fig.py` | `trades.png` (2-panel ΔP/ΔR) — **run after** `report_figs_v2.py` |
| `report_fig_tolerance.py` | `tolerance.png` |
| `report_fig_interaction.py` | `interaction.png` |
| `bylength_fig.py` | `bylength.png` (5 ladder models incl. the gated adapter) |
| `datascale_plot.py` | `datascale_curve.png` |
| `datascale_tol_plot.py` | `datascale_tolerance.png` |
| `sim_fig_report.py` | `similarity.png` |
| `ladder_fig.py` | `figures/ladder/` — `ladder.png`, `ladder_tol.png` + build-up frames `f1..f6` |

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

## Gated-adapter ablation pipeline (isolates the gated seq-adapter)
| script | role |
|---|---|
| `train_adapter_ablation.py` | trains `runs/2026_esmc6b_adapterNNN_seqonly` = zeroctrl recipe + `--gated_seq_only` (seq adapter, no struct). Usage: `<seq_proj> <out_dir> [epochs]` |
| `eval_adapter256.py` | `adapter256_perprotein_2026.csv` + paired adapter Δ vs orange |
| `eval_adapter256_tol.py` | `adapter256_tol_perprotein.csv` (tolerance curve) |
| `eval_adapter256_seg.py` | `adapter256_seg_true.csv` + `adapter256_seg_pred.csv` (per-segment lengths, for bylength) |
| `esm2_boundary_infer.py` | ESM-2 boundary-head Δ (baseline_esm2 vs esm2_boundary) |
