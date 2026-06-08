# Experiment Architecture Documentation

**Scope:** All Table-1 and Table-2 variants from `texs/Overleaf/experiments.tex`
(as mapped in `analysis/table_verification.md` and `analysis/canonical_metrics.md`).

**Sources:** `runs/*/config.json`, `src/models/crf_models.py`, `src/train_loop_crf.py`.

---

## Common Backbone

All variants use the same **LSTM-CNN-CRF backbone** unless stated otherwise.

**LSTMCNN feature extractor** (`src/models/lstm_cnn.py`, class `LSTMCNN`):
1. `Dropout1d(p=0.1)` on the raw embedding channels.
2. `Conv1d(C_in → 32 filters, kernel=3, padding=1)` + ReLU.
3. `Dropout1d(p=0.1)`.
4. Bidirectional LSTM: input=32, hidden=64 → output 128 per position.
5. `Conv1d(128 → 64 filters, kernel=5, padding=2)` + ReLU.

Output: `[B, L, 64]` contextual features per residue.

**Emission head:** `Linear(64 → num_labels)` where num_labels=3 (None / Peptide / Propeptide).

**Multistate CRF** (101 states for 3-label models):
- States 0 = None; states 1–50 = peptide progress (strict grammar enforcing length 5–50); states 51–100 = propeptide progress (same grammar).
- The 3-label emission vector is broadcast into 101 states by repeating logit 1 into all peptide states and logit 2 into all propeptide states.
- Trained with negative CRF log-likelihood (mean reduction).
- Best checkpoint = max of (F1_pep + F1_prop)/2 on validation set.

**All common hyperparameters** (unless a variant explicitly overrides):
- `epochs=100`, `batch_size=48`, `lr=1e-4`, `optimizer=Adam`
- `dropout=0.1`, `conv_dropout=0.1`, `kernel_size=3`, `num_filters=32`, `hidden_size=64`
- `label_type=multistate_with_propeptides`
- `amp=True, amp_dtype=bf16` (some runs use `amp=False`)

---

## Table 1: Architectural Changes

All Table-1 variants use **ESM2 embeddings** (`embedding_dim=1280`,
`embeddings_dir=data/uniprot_2022/embeddings/embeddings_esm2` or equivalents).

---

### T1-1 · ESM2 (baseline)

**Folder:** `runs/train_run_esm2`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Verbatim backbone described above. Raw ESM2 embeddings fed directly into `LSTMCNN`. No projector, no auxiliary signal, no secondary losses.
**Key config:** `embedding_dim=1280`

**One-line summary:** Raw ESM2 (1280-d) → LSTMCNN (conv32/biLSTM64/conv64) → linear → 101-state CRF; Adam lr=1e-4; no projector.

---

### T1-2 · ESM2 + telescopic CRF

**Folder:** `runs/esm2_telescoping_segmental`
**Model class:** `LSTMCNNCRFTelescopingSegmental` (`model=lstmcnncrf_telescoping_segmental`)
**Architecture:** Same LSTMCNN backbone but the CRF uses **strict length states** (1 per length step) instead of the compressed skip-connection grammar. Two extra components replace the standard emission broadcasting:
  - `features_to_emissions`: Linear(64 → 3) — coarse per-label logit (None/Pep/Prop), same as baseline.
  - `position_head`: `LayerNorm + Dropout + Linear(64 → 2)` → sigmoid, predicts relative position within segment (one value per segment label). Zero-initialized.
  - **SpanScore** is computed for every (end, length) pair as `sum_of_coarse_logits + position_scale * sum_of_position_scores`. Delta emissions (telescoping) are derived from successive SpanScore differences.
  - CRF constraints: strict sequential grammar (state k → k+1 within segment, jump back to 0 or other-branch start after min_len).
  - `segmental_max_len=50`, `segmental_min_len=1`, `position_score_mode=neg_abs`, `position_score_tau=0.25`, `position_score_scale=0.25`, `relative_position_loss_lambda=0.0` (no auxiliary position loss in this run).
  - Decoded strict paths are converted back to legacy compressed states for metric compatibility.
**Key config:** `embedding_dim=1280` (using `data/embeddings_esm2` path, different from baseline's `data/uniprot_2022/embeddings/embeddings_esm2` — these may differ in coverage; see experiments verification notes).

**One-line summary:** ESM2 → LSTMCNN → span-score telescoping CRF with strict length states and position-progress head (zero-initialized, lambda=0).

---

### T1-3 · ESM2 + Aho emission fusion

**Folder:** `runs/esm2_aho_emission_fusion`
**Model class:** `LSTMCNNCRFAhoEmissionFusion` (`model=lstmcnncrf_aho_emission_fusion`)
**Architecture:** Input is a concatenated `[ESM2 | Aho features]` tensor (1280+76=1356 channels).
  - **Sequence branch:** raw ESM2 channels → plain LSTMCNN → `Linear(64 → 3)` base emissions.
  - **Aho branch (late fusion):** Aho features (76-d per residue, sparse dictionary hits) → `LayerNorm + Dropout + Linear(76 → 3)` → per-label additive bias. Zero-initialized (starts as baseline).
  - `aho_hidden_size=0` (direct linear, no hidden layer in the Aho branch).
  - Final emission = base + aho_bias (per-label scales are all 1.0 in this run).
  - `aho_branch_dropout=0.1` (whole-sample Aho branch zeroed with p=0.1 during training).
**Key config:** `embedding_dim=1356`, `seq_input_size=1280`, `residue_input_size=76`, `aho_hidden_size=0`, `struct_input_size=20` (struct present in concat but not used by this model class — struct channels are part of the Aho concat block).

**One-line summary:** ESM2 branch (LSTMCNN, unchanged) + Aho-76d late-fusion linear head adding per-label emission bias; zero-initialized; no hidden layer.

---

### T1-4 · ESM2 + (Aho → hidden layer 32) emission fusion

**Folder:** `runs/esm2_aho_emission_fusion_h32`
**Model class:** `LSTMCNNCRFAhoEmissionFusion` (`model=lstmcnncrf_aho_emission_fusion`)
**Architecture:** Identical to T1-3 except `aho_hidden_size=32` — the Aho branch is a two-layer MLP: `LayerNorm + Dropout + Linear(76 → 32) + GELU + Dropout + Linear(32 → 3)`, still zero-initialized at the last layer.
**Key config:** `embedding_dim=1356`, `aho_hidden_size=32`

**One-line summary:** Same as T1-3 but Aho head has an intermediate hidden layer (76→32→3 with GELU) instead of direct linear.

---

### T1-5 · ESM2 + Aho hidden state fusion

**Folder:** `runs/esm2_aho_mid_fusion_raw_m64`
**Model class:** `LSTMCNNCRFAhoMidFusion` (`model=lstmcnncrf_aho_mid_fusion`)
**Architecture:** Input: `[ESM2 | Aho]` (1356 channels).
  - **Sequence branch:** ESM2 → LSTMCNN → features h_i (64-d per position).
  - **Aho encoder:** `LayerNorm + Dropout` over raw Aho features (no projection since `aho_hidden_size=0`); output a_i = 76-d normalized Aho.
  - **Mid-fusion head:** concatenate `[h_i; a_i]` (64+76=140-d) → `LayerNorm + Dropout + Linear(140 → 64, hidden) + GELU + Dropout + Linear(64 → 3)` — additive correction to base emission. Zero-initialized at last layer.
  - This means the Aho correction is conditioned on both the neural context and dictionary hits.
  - `aho_mid_hidden_size=64`, `aho_hidden_size=0` (encoder is LayerNorm only).
**Key config:** `embedding_dim=1356`, `aho_mid_hidden_size=64`

**One-line summary:** ESM2 → LSTMCNN gives h_i; Aho features (76-d, LayerNorm only) give a_i; mid-fusion MLP([h_i, a_i], hidden=64) adds emission correction; zero-initialized.

---

### T1-6 · ESM2 + Aho hidden state fusion only peptides

**Folder:** `runs/esm2_aho_mid_fusion_raw_m64_pep_only`
**Model class:** `LSTMCNNCRFAhoMidFusion` (`model=lstmcnncrf_aho_mid_fusion`)
**Architecture:** Same as T1-5 but Aho label scales set so that the Aho correction is **applied only to the Peptide label** and zeroed for None and Propeptide:
  - `aho_none_scale=0.0`, `aho_pep_scale=1.0`, `aho_propep_scale=0.0`
  - Intended to test whether Aho dictionary hits (which target mature peptides) should only inform the peptide emission, not propeptide.
**Key config:** `embedding_dim=1356`, `aho_none_scale=0.0`, `aho_pep_scale=1.0`, `aho_propep_scale=0.0`

**One-line summary:** Identical to T1-5 (mid-fusion, hidden=64) but Aho correction multiplied by zero for None and Propeptide labels — peptide-only Aho influence.

---

### T1-7 · ESM2 + Aho сигнал добавляется к CRF переходам (Aho transition bias)

**Folder:** `runs/esm2_aho_transition_bias_sparse_trainable_zero`
**Model class:** `lstmcnncrf_aho_transition_bias_sparse` — this class no longer exists in the codebase; the checkpoint cannot be reloaded (`strict=True` fails). MCC/AUC are N/A. (Not to be confused with the separate `LSTMCNNCRFAhoStateBias` / `lstmcnncrf_aho_state_bias` class used in unreferenced run `esm2_aho_state_bias_pep_boundary_010`.)
**Architecture (as stored in checkpoint):** Input `[ESM2 | Aho]` (1356 channels). Sequence branch: ESM2 → LSTMCNN → coarse 3-label emissions broadcast to 101 states. Then, Aho-derived scalar biases are added **directly to CRF state emissions** (not to coarse labels): specific Aho features (pep.start_decay, pep.inside, pep.end_decay, propep equivalents) are looked up by name from `aho_feature_names_file`, weighted by per-role learnable biases (`aho_transition_bias_trainable=True`), and added to the start/inside/end states of each branch. The biases are initialized to zero — model starts as baseline, learns when Aho boundary signals help.
**Key config:** `aho_transition_boundary_feature=decay`, `aho_transition_bias_trainable=True`, all init biases=0.0. Model class `lstmcnncrf_aho_transition_bias_sparse` is no longer in codebase.

**One-line summary:** ESM2 LSTMCNN baseline + trainable Aho-feature state biases injected into CRF state emissions at start/inside/end positions (decay boundary features); model unrecoverable for fresh inference.

---

### T1-8 · ESM2 + Aho early fusion (concat with esm)

**Folder:** `runs/esm2_aho_tribranch`
**Model class:** `LSTMCNNCRFTriBranchResidual` (`model=lstmcnncrf_tribranchresidual`)
**Architecture:** Three-branch gated residual projector before the LSTMCNN backbone.
  - Input: `[ESM2 (1280) | Aho (76) | struct (0)]` = 1356 channels (struct_input_size=0, so no structural branch).
  - **ESM2 branch (main):** `EmbeddingProjector(1280 → 256)` = `LayerNorm + Dropout + Linear + GELU + Dropout`.
  - **Residue/Aho branch:** `EmbeddingProjector(76 → 16)` → gated residual added to main branch. Gate is `sigmoid(Linear([seq_proj; residue_proj]))`, init'd to near-zero. Residual scale = 0.05.
  - Output: 256-d fused representation → LSTMCNN (input=256).
  - This is "early fusion": Aho features are mixed into the ESM2 representation *before* the LSTM/CNN backbone, unlike T1-3/T1-5 where fusion happens after.
  - `residue_branch_dropout=0.3` (whole Aho branch zeroed during training with p=0.3).
**Key config:** `embedding_dim=1356`, `seq_input_size=1280`, `struct_input_size=0`, `residue_input_size=76`, `seq_proj_size=256`, `residue_proj_size=16`, `residue_residual_scale=0.05`, `residue_gate_bias=-2.5`

**One-line summary:** Three-branch projector fuses Aho features into ESM2 representation before LSTMCNN (early, gated residual, scale=0.05, gate bias=-2.5); 256-d main branch.

---

### T1-9 · ESM2 + доп. лосс разрезов к ближайшей границе (auxiliary bond/cleavage loss)

**Folder:** `runs/esm2_bond_loss_soft_l005_w5_tau15`
**Model class:** Old class `lstmcnncrf_bond_loss` — no longer exists in codebase as `lstmcnncrf_bond_loss` (current codebase has `lstmcnncrf_boundary_bond_loss`). Checkpoint unrecoverable for fresh inference. MCC/AUC are N/A.
**Architecture (CORRECTED — grounded in the actual checkpoint, not the current class):** ESM2 (1280-d) → LSTMCNN → baseline CRF emissions, plus ONE auxiliary head. The trained checkpoint's `state_dict` modules are exactly `[feature_extractor, features_to_emissions, crf, bond_head]` — there is **no** boundary-state head. So:
  - `bond_head`: a 3-layer MLP (`bond_head.0/2/5` — Linear→act→Linear→act→Linear) predicting a per-adjacent-pair cleavage logit. (This differs from the current `lstmcnncrf_boundary_bond_loss` class, whose head is `bond_head.net.*` plus a `boundary_to_state` head — which is why the checkpoint won't load into current code and is unrecoverable.)
  - **Auxiliary soft bond loss only:** soft targets spread around true cleavage sites via exponential decay (window=5, tau=1.5), weight `bond_loss_lambda=0.05`. The bond head affects only the *training loss*; it does NOT modify CRF emissions at inference. So the decoded predictions come from the plain baseline emission path — this run is architecturally the baseline trained with an extra cleavage-aux loss.
  - Input: ESM2 only (1280-d); Aho-like config fields are template carry-over, unused.
**Key config:** `bond_loss_lambda=0.05`, `bond_soft_window=5`, `bond_soft_tau=1.5`, `bond_soft_mode=exp`, `bond_positive_weight=10.0`.

**One-line summary:** baseline ESM2 LSTMCNN-CRF + a training-only auxiliary soft cleavage-site bond loss (λ=0.05, exp window=5, τ=1.5) via a standalone bond_head MLP; no boundary-state head; checkpoint unrecoverable for fresh inference → MCC/AUC N/A.

---

### T1-10 · ESM2 c AdamW оптимизатором

**Folder:** `runs/train_run_esm2_adamw`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same `LSTMCNNCRF` baseline as T1-1. Despite the name "AdamW", the current `train_loop_crf.py` uses `Adam` (no weight decay). The architectural difference from the baseline: this run uses `epochs=60` instead of 100, and `amp=True` (bf16 AMP). The config also carries many unused auxiliary parameters (bond head, boundary head, Aho fields — all at zero/disabled defaults), suggesting this config was adapted from a later template. In practice it is the same architecture as T1-1 trained with AMP and fewer epochs.
**Key config:** `embedding_dim=1280`, `epochs=60`, `amp=True`

**One-line summary:** Same architecture as baseline (LSTMCNNCRF, ESM2 1280-d) but trained with bf16 AMP and 60 epochs instead of 100; optimizer label misleading (still Adam in code).

---

## Table 2: Embedding Generators

All Table-2 variants use the **same LSTMCNNCRF architecture** (`model=lstmcnncrf`) unless otherwise noted (the ESM2+3Di and AFT rows use specialized fusion models). What changes is the **embedding source** (and thus embedding dimension and model architecture for fusion variants).

---

### T2-1 · ESM2

Same as T1-1 (shared folder `runs/train_run_esm2`). See above.

**One-line summary:** ESM2 (1280-d) → LSTMCNNCRF; Adam; same as Table 1 baseline.

---

### T2-2 · ESM2+residue features (ESM2+ below)

**Folder:** `runs/train_run_esm2_plus`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same LSTMCNNCRF backbone. Input: precomputed concatenation of ESM2 (1280-d) + 10 per-residue biochemical features (e.g., one-hot amino acid class, surface accessibility, secondary structure indicators). Total `embedding_dim=1290`. The extra 10 features are concatenated at the embedding level before the CNN/LSTM; no projector separates them.
**Key config:** `embedding_dim=1290`, `embeddings_dir=.../embeddings_esm2_plus`

**One-line summary:** ESM2 + 10 biochemical residue features concatenated (1290-d total) → LSTMCNNCRF; no separate projector; K=10.

---

### T2-3 · ESM-C

**Folder:** `runs/train_run_esmc_600m`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Identical architecture to baseline. Input: ESM-C (600M) embeddings, `embedding_dim=1152`. AMP enabled (bf16).
**Key config:** `embedding_dim=1152`, `embeddings_dir=.../embeddings_esmc`

**One-line summary:** ESM-C 600M (1152-d) → LSTMCNNCRF; same architecture as ESM2 baseline.

---

### T2-4 · ESM-C 6B

**Folder:** `runs/esmc_6b`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same LSTMCNNCRF architecture. Input: ESM-C 6B embeddings, `embedding_dim=2560`. Larger embedding dimension; config carries many unused auxiliary parameters from later template.
**Key config:** `embedding_dim=2560`, `embeddings_dir=.../embeddings_esmc6b/`

**One-line summary:** ESM-C 6B (2560-d) → LSTMCNNCRF; otherwise identical architecture to ESM2 baseline.

---

### T2-5 · ProstT5

**Folder:** `runs/train_run_prostt5`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Identical architecture to baseline. Input: ProstT5 embeddings, `embedding_dim=1024`. AMP enabled.
**Key config:** `embedding_dim=1024`, `embeddings_dir=.../embeddings_prostt5`

**One-line summary:** ProstT5 (1024-d sequence embeddings) → LSTMCNNCRF; same architecture as ESM2 baseline.

---

### T2-6 · ProstT5+residue features

**Folder:** `runs/train_run_prostt5_plus`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same as T2-2 but with ProstT5 embeddings. Concatenation of ProstT5 (1024-d) + 10 per-residue features = 1034-d total. AMP enabled.
**Key config:** `embedding_dim=1034`, `embeddings_dir=.../embeddings_prostt5_plus`

**One-line summary:** ProstT5 + 10 residue features (1034-d) → LSTMCNNCRF; same as ESM2+ but with ProstT5 base.

---

### T2-7 · (ProstT5 3DI + ESM2) proj.

**Folder:** `runs/train_run_esm2+3di_proj`
**Model class:** `LSTMCNNCRFSplitProjector` (`model=lstmcnncrf_projector_split`)
**Architecture:** Input: precomputed concatenation of ESM2 (1280-d) + ProstT5-3Di one-hot (20-d) = 1300-d.
  - **Split projector:** ESM2 and 3Di are projected separately before concatenation:
    - ESM2: `EmbeddingProjector(1280 → 256)` = `LayerNorm + Dropout + Linear + GELU + Dropout`.
    - 3Di: `EmbeddingProjector(20 → 64)`.
    - Outputs concatenated → 320-d input to LSTMCNN.
  - LSTMCNN processes the 320-d concatenated projection.
  - No gating — both branches contribute equally to the joint representation.
  - `amp=True`
**Key config:** `embedding_dim=1300`, `seq_input_size=1280`, `struct_input_size=20`, `seq_proj_size=256`, `struct_proj_size=64`

**One-line summary:** ESM2 (→256) and ProstT5-3Di-20d (→64) projected separately, concatenated to 320-d, then LSTMCNNCRF; no gating between branches.

---

### T2-8 · (ProstT5 3DI + ESM2) proj.gated.

**Folder:** `runs/train_run_esm2+3di_proj_gated`
**Model class:** `LSTMCNNCRFGated3DiResidual` (`model=lstmcnncrf_gated3diresidual`)
**Architecture:** Input: same 1300-d concatenation. Uses **gated residual fusion** instead of concatenation:
  - ESM2 projected to 256-d (main branch).
  - 3Di projected to 32-d, then linearly mapped to 256-d.
  - A per-position gate vector is computed from `sigmoid(Linear([seq_proj; struct_proj]))` (init bias = -2.0, so gate ≈ 0.12 initially).
  - Fused = `LayerNorm(seq_proj + residual_scale * gate * struct_up)` where `residual_scale=0.2`.
  - Output: 256-d fused ESM2-dominant representation → LSTMCNN.
  - `struct_branch_dropout=0.3` during training.
**Key config:** `seq_proj_size=256`, `struct_proj_size=32`, `gated_residual_scale=0.2`, `gated_gate_bias=-2.0`, no `struct_conv_kernel` (no conv on structural branch)

**One-line summary:** ESM2 (→256, main) + 3Di (→32, gated residual, scale=0.2, gate_bias=-2.0); no conv on structural branch; output 256-d → LSTMCNNCRF.

---

### T2-9 · (ProstT5 3DI + ESM2) proj.gated.conv.

**Folder:** `runs/train_run_esm2+3di_proj_gated_conv`
**Model class:** `LSTMCNNCRFGated3DiResidualConv` (`model=lstmcnncrf_gated3diresidual_conv`)
**Architecture:** Extends T2-8 by adding a **Conv1d on the structural branch** before gating:
  - 3Di projected to 16-d (smaller), then `Conv1d(16 → 16, kernel=5)` + GELU + Dropout to capture local structural patterns.
  - Then gated residual as in T2-8: gate computed from `[seq_proj; struct_conv_out]`, added with scale=0.1 (smaller than T2-8).
  - `gated_gate_bias=-2.5` (gate ≈ 0.076 initially, tighter than T2-8).
  - `struct_branch_dropout=0.5` (stronger dropout on structural branch).
**Key config:** `seq_proj_size=256`, `struct_proj_size=16`, `struct_conv_kernel=5`, `gated_residual_scale=0.1`, `gated_gate_bias=-2.5`

**One-line summary:** Like T2-8 but 3Di branch uses Conv1d(k=5) before gating, smaller proj (16-d), smaller residual scale (0.1), tighter gate (-2.5); 50% struct branch dropout.

---

### T2-10 · (ProstT5 3DI + ESM2+) proj.gated.conv. — DROPPED

This row was present in the LaTeX source but has **no matching run folder** in `runs/`. Removed from tables in commit a45f104.

**One-line summary:** Row dropped — no backing experiment found; likely intended as ESM2+ (1290-d) + 3Di with the proj.gated.conv. fusion, but the run is lost.

---

### T2-11 · AFTK all, no filter

**Folder:** `runs/train_run_aft`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same LSTMCNNCRF baseline architecture. Input: AlphaFold Toolkit (AFT) embeddings using all AFT channels without pLDDT filtering. `embedding_dim=563`. NOTE: AFT runs have unreliable fresh inference metrics because the embedding files were remapped during repo reorg (see experiments verification memory). Train-time P/R/F1 are authoritative; MCC/AUC are N/A.
**Key config:** `embedding_dim=563`, `embeddings_dir=.../embeddings_aft`

**One-line summary:** AFT all-channel embeddings (563-d, no filter) → LSTMCNNCRF; same architecture as ESM2 baseline but different embedding.

---

### T2-12 · AFTK only single, no filter

**Folder:** `runs/train_run_aft_single`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same LSTMCNNCRF baseline. Input: AFT single-chain embeddings only (no pairwise), `embedding_dim=384`. AFT drift issue applies.
**Key config:** `embedding_dim=384`, `embeddings_dir=.../embeddings_aft_single`

**One-line summary:** AFT single-chain only embeddings (384-d, no filter) → LSTMCNNCRF; same architecture.

---

### T2-13 · AFTK all w/o lddt, no filter

**Folder:** `runs/train_run_aft_no_lddt`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same LSTMCNNCRF baseline. Input: AFT all-channel embeddings with pLDDT column removed, `embedding_dim=513`. AFT drift issue applies.
**Key config:** `embedding_dim=513`, `embeddings_dir=.../embeddings_aft_no_lddt`

**One-line summary:** AFT all-channel without pLDDT column (513-d, no filter) → LSTMCNNCRF; same architecture.

---

### T2-14 · AFTK all, >70% avg pLDDT

**Folder:** `runs/train_run_aft_plddt70`
**Model class:** `LSTMCNNCRF` (`model=lstmcnncrf`)
**Architecture:** Same LSTMCNNCRF baseline. Input: AFT all-channel embeddings but proteins filtered to >70% average pLDDT (222,308 residues vs 316,314 for unfiltered AFT), `embedding_dim=563`. AFT drift issue applies.
**Key config:** `embedding_dim=563`, `embeddings_dir=.../embeddings_aft_plddt70`

**One-line summary:** AFT all-channel embeddings (563-d), proteins filtered to >70% avg pLDDT → LSTMCNNCRF; same architecture.

---

### T2-15 · ESM2+(AFTK all, no filter) pr.gt.conv

**Folder:** `runs/train_run_esm2_aft`
**Model class:** `LSTMCNNCRFGated3DiResidualConv` (`model=lstmcnncrf_gated3diresidual_conv`)
**Architecture:** Same proj.gated.conv. fusion as T2-9 but with **AFT all-channel (563-d)** as the structural branch instead of ProstT5 3Di.
  - Input: `[ESM2 (1280) | AFT all (563)]` = 1843-d concatenated.
  - ESM2 projected to 256-d; AFT projected to 64-d, then Conv1d(64→64, k=5) + GELU + Dropout; gated residual added to ESM2 branch (scale=0.2, gate_bias=-2.5).
  - `struct_branch_dropout=0.3`; `amp=True`.
  - AFT drift issue applies to MCC/AUC.
**Key config:** `embedding_dim=1843`, `seq_input_size=1280`, `struct_input_size=563`, `struct_proj_size=64`, `gated_residual_scale=0.2`, `gated_gate_bias=-2.5`

**One-line summary:** ESM2 (→256, main) + AFT all-channel (→64, Conv1d k=5, gated residual scale=0.2) — same fusion as T2-9 but structural branch = AFT 563-d.

---

### T2-16 · ESM2+(AFTK only single no filter) pr.gt.conv

**Folder:** `runs/train_run_esm2_aft_single_gated`
**Model class:** `LSTMCNNCRFGated3DiResidualConv` (`model=lstmcnncrf_gated3diresidual_conv`)
**Architecture:** Same gated conv fusion. Structural branch: AFT single-chain (384-d).
  - Input: `[ESM2 (1280) | AFT single (384)]` = 1664-d. Structural proj → 64-d, Conv1d(k=5), gated residual.
**Key config:** `embedding_dim=1664`, `struct_input_size=384`, `struct_proj_size=64`, `gated_residual_scale=0.2`

**One-line summary:** ESM2 + AFT single-chain (384-d) structural branch; same gated conv fusion as T2-15.

---

### T2-17 · ESM2+(AFTK only pair no filter) pr.gt.conv

**Folder:** `runs/train_run_esm2_aft_pair_gated`
**Model class:** `LSTMCNNCRFGated3DiResidualConv` (`model=lstmcnncrf_gated3diresidual_conv`)
**Architecture:** Same gated conv fusion. Structural branch: AFT pairwise-only (128-d, obtained by subtracting single from all).
  - Input: `[ESM2 (1280) | AFT pair (128)]` = 1408-d. Structural proj → 64-d, Conv1d(k=5), gated residual.
**Key config:** `embedding_dim=1408`, `struct_input_size=128`, `struct_proj_size=64`, `gated_residual_scale=0.2`

**One-line summary:** ESM2 + AFT pairwise-only (128-d) structural branch; same gated conv fusion architecture.

---

### T2-18 · ESM2+(AFTK all w/o lddt no filter) pr.gt.conv

**Folder:** `runs/train_run_esm2_aft_no_lddt_gated`
**Model class:** `LSTMCNNCRFGated3DiResidualConv` (`model=lstmcnncrf_gated3diresidual_conv`)
**Architecture:** Same gated conv fusion. Structural branch: AFT all without pLDDT (513-d).
  - Input: `[ESM2 (1280) | AFT no-lddt (513)]` = 1793-d. Structural proj → 64-d, Conv1d(k=5), gated residual.
  - AFT drift issue applies to MCC/AUC.
**Key config:** `embedding_dim=1793`, `struct_input_size=513`, `struct_proj_size=64`, `gated_residual_scale=0.2`

**One-line summary:** ESM2 + AFT all-channel without pLDDT (513-d); same gated conv fusion as T2-15.

---

### T2-19 · ESM2+(AFTK all, >70% avg pLDDT) pr.gt.conv

**Folder:** `runs/train_run_esm2_aft_plddt70`
**Model class:** `LSTMCNNCRFGated3DiResidualConv` (`model=lstmcnncrf_gated3diresidual_conv`)
**Architecture:** Same gated conv fusion. Structural branch: AFT all-channel pLDDT-filtered (563-d, same dim as unfiltered all but fewer proteins covered).
  - Input: `[ESM2 (1280) | AFT plddt70 (563)]` = 1843-d. Structural proj → 64-d, Conv1d(k=5), gated residual.
  - Training set for AFT proteins reduced to those with >70% avg pLDDT.
**Key config:** `embedding_dim=1843`, `struct_input_size=563`, `struct_proj_size=64`, `gated_residual_scale=0.2`

**One-line summary:** ESM2 + AFT all-channel pLDDT>70% filtered (563-d); same gated conv fusion as T2-15; reduced training coverage.

---

## Architecture Taxonomy

| Model key | Description | Table appearances |
|:--- |:--- |:--- |
| `lstmcnncrf` | Raw embeddings → LSTMCNN → CRF | T1-1, T1-10, T2-1..T2-6, T2-11..T2-14 |
| `lstmcnncrf_telescoping_segmental` | LSTMCNN + span-score telescoping CRF | T1-2 |
| `lstmcnncrf_aho_emission_fusion` | LSTMCNN + Aho late-fusion (linear/MLP emission bias) | T1-3, T1-4 |
| `lstmcnncrf_aho_mid_fusion` | LSTMCNN + Aho mid-fusion (concat h_i + a_i → MLP correction) | T1-5, T1-6 |
| `lstmcnncrf_aho_transition_bias_sparse` | LSTMCNN + Aho state-level CRF emission bias (unrecoverable) | T1-7 |
| `lstmcnncrf_tribranchresidual` | Three-branch gated projector (ESM2+Aho+struct) → LSTMCNN | T1-8 |
| `lstmcnncrf_bond_loss` | LSTMCNN + auxiliary soft cleavage loss (old class, unrecoverable) | T1-9 |
| `lstmcnncrf_projector_split` | ESM2 + 3Di → split projector → concatenate → LSTMCNNCRF | T2-7 |
| `lstmcnncrf_gated3diresidual` | ESM2 (main) + struct (gated residual, no conv) | T2-8 |
| `lstmcnncrf_gated3diresidual_conv` | ESM2 (main) + struct (gated residual, with Conv1d k=5) | T2-9, T2-15..T2-19 |

---

## VERIFY Checklist

Items that should be double-checked against the code / metrics before finalizing:

1. **T1-2 embedding path discrepancy:** `esm2_telescoping_segmental` uses `data/embeddings_esm2` (not `data/uniprot_2022/embeddings/embeddings_esm2`). Confirm whether these are the same files or differ in coverage. If different, this run's embeddings may not match the other ESM2 runs.

2. **T1-7 unrecoverable class:** Verify that `lstmcnncrf_aho_transition_bias_sparse` is genuinely absent from all git history (not just the current HEAD), confirming the model is permanently unrecoverable for MCC/AUC.

3. **T1-9 bond_loss vs boundary_bond_loss:** `esm2_bond_loss_soft_l005_w5_tau15` uses `model=lstmcnncrf_bond_loss` which is not in current `train_loop_crf.py` (current name is `lstmcnncrf_boundary_bond_loss`). The architecture description above is inferred from the bond-loss config fields + the `LSTMCNNCRFBoundaryBondLoss` docstring. Confirm that the old `lstmcnncrf_bond_loss` class was equivalent but lacked the `boundary_to_state` head (experiments verification notes say it had `bond_head.0.*` but no boundary head).

4. **T1-10 "AdamW" label:** The run name says AdamW but `train_loop_crf.py` only has `Adam`. Confirm no external optimizer override was used at training time (e.g., a different version of `run.py` at train time that did pass `AdamW`).

5. **T2-2, T2-6 residue features:** Confirm what the 10 extra features are in `embeddings_esm2_plus` and `embeddings_prostt5_plus`. The embedding_dim difference (1290-1280=10 and 1034-1024=10) confirms 10 features but the specific features (AA class, secondary structure, SASA, etc.) should be documented from the embedding-generation scripts.

6. **T2-11..T2-14 AFT drift:** MCC/AUC are N/A for most AFT-only runs due to embedding remapping. Confirm which specific AFT runs have MCC/AUC available vs N/A in `canonical_metrics.md` (T2-14 `aft_plddt70` has MCC/AUC; T2-11, T2-12, T2-13 do not).

7. **T2-17 pair-only dim:** The AFT pair embedding is 128-d. Confirm this is `aft_all(563) - aft_single(384) - plddt_col(1)` = 178 residual, OR that there is a dedicated precomputed pairwise-only embedding with 128-d. The config `struct_input_size=128` is authoritative.

8. **Aho features (76-d):** The Aho input size `residue_input_size=76` and `struct_input_size=20` appear together in the aho-fusion configs (the 76 includes Aho dictionary features). Confirm what makes up the 76 channels vs the 20 structural channels in `data/embeddings_esm2_aho_train012/`.

---

## C. Best combined configuration — ESM-C 6B × boundary/bond (`esmc6b_boundary_bond`)

A deliberate cross of Section A and Section B rather than a sweep entry: the best
**embedding** by residue-level signal (ESM-C 6B, 2560-d) fed into the best
boundary-aware **architecture** (`lstmcnncrf_boundary_bond_loss`).

- **Embedding:** ESM-C 6B per-residue, 2560-d (`embeddings_esmc6b`). On its own (plain
  `lstmcnncrf`) it has the highest recall in Table 2 but the lowest precision, so its
  ±3 F1 lags the ESM2 baseline.
- **Architecture:** `LSTMCNNCRFBoundaryBondLoss` (= baseline LSTMCNN→CRF + a learned
  start/inside/end **boundary-state emission head** added to the CRF emissions, +
  an auxiliary **soft cleavage-site bond loss** at training, defaults λ=0.02 / window 5
  / τ 1.5 / pos-weight 10). `embedding_dim=2560`; 365k trainable params. NB: this is
  the *current* `boundary_bond_loss` class (boundary head + bond aux loss) — distinct
  from the older, checkpoint-only bond-only variant `esm2_bond_loss_soft` (T1-9).
- **Result (TEST, fp32, drift 0.000):** F1 0.657 / P 0.714 / R 0.609 / MCC 0.765 —
  best F1 and MCC in the project (ESM2 baseline 0.607/0.640/0.578/0.750; ESM-C 6B
  baseline 0.579/0.570/0.590/0.758). Also best on the HOMO slice (F1 0.548 / MCC 0.747
  vs baseline 0.460 / 0.693).
- **Why it works:** the boundary head sharpens cleavage calls, so it converts ESM-C 6B's
  abundant-but-fuzzy residue signal into **precision (+0.14, 0.570→0.714)** with recall
  held. The same architecture barely moved the already-balanced ESM2 baseline (F1
  0.606≈0.607) — the benefit is conditional on the embedding's P/R profile.
- **Caveat:** single seed (training variance ~±0.02; gain ≈2.5×). Full writeup +
  corrected-metric confirmation in `texs/error_analysis/combine_best.md`.

---

## VERIFY items — RESOLVED (checked against code + actual tensor shapes)

Ground-truth `.pt` shapes (first protein, L=267): esm2 `(L,1280)`, esm2_plus `(L,1290)`, esm2_3di `(L,1300)`, aft `(L,563)`, aft_single `(L,384)`, aft_pair `(L,128)`, esm2_aho_train012 `(L,1356)`.

1. **(#1 telescoping path)** RESOLVED — `esm2_telescoping_segmental` reproduced train-time metrics EXACTLY (Δ=0) on re-inference, so its `data/embeddings_esm2` is the same content as the baseline's esm2 embeddings. Comparable.
2. **(#2 transition_bias)** CONFIRMED unrecoverable — `git log --all -S 'aho_transition_bias'` over `*.py` is empty; the class was never committed. `LSTMCNNCRFAhoStateBias`/`lstmcnncrf_aho_state_bias` IS a different, present class (used only by the untabled run `esm2_aho_state_bias_pep_boundary_010`).
3. **(#3 bond_loss)** RESOLVED & doc corrected — the trained checkpoint has modules `[feature_extractor, features_to_emissions, crf, bond_head]` with a 3-layer `bond_head.0/2/5` MLP and **no boundary-state head**. The earlier description (boundary head + `BondBoundaryHead`) was the *current* class, not the trained one. T1-9 above is fixed: it's the baseline + a training-only cleavage aux loss.
4. **(#4 AdamW)** RESOLVED — config has NO optimizer field; `AdamW` never appears in git history of `*.py`. The only recorded difference from baseline is `epochs=60` (vs 100). Treat as "baseline, fewer epochs"; the AdamW label is unverified/misleading.
5. **(#5 residue features)** RESOLVED — `make_embeddings_esm2_with_features.py` defines exactly K=10 deterministic per-residue features from the AA sequence only: hydrophobicity, charge7 (K/R=+1, D/E=−1, H=+0.1), is_polar, …, net_charge_w9 (windowed charge sum). Concatenated to the RIGHT of the embedding (1280→1290, 1024→1034). NOT PSSM/SASA/secondary-structure.
6. **(#6 AFT drift)** Per user: AFT embeddings were intentionally NOT computed for all proteins — partial coverage is the expected/original format, not a remap bug. Fresh MCC/AUC stay N/A where drift>0.015 (conservative); P/R/F1 are the authoritative train-time values.
7. **(#7 aft_pair dim)** RESOLVED — `aft_pair` is a dedicated precomputed embedding stored at **128-d** (actual tensor shape `(L,128)`), not a runtime subtraction of aft_all−aft_single.
8. **(#8 Aho channels)** RESOLVED (dim arithmetic) — `embeddings_esm2_aho_train012` is `(L,1356)` = ESM2 1280 + **76 Aho channels**, matching `residue_input_size=76` in the aho-fusion configs. (Exact per-channel semantics — motif hit/score layout — see `make_embeddings_aho.py`; not critical for the architecture description.)
