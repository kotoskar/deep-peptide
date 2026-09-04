# Pending review findings — 2026-09-04

Two adversarial review passes were **stopped part-way** to save budget. What they had already
produced is recorded here verbatim; **nothing below has been applied to the paper or the code**.

* The paper fact-check (`main.tex`) finished all six reading lenses but ran **no adjudication**,
  so every item is a *candidate*: the reading agent believed it, a refuter never tried to kill it.
  Verify each against the artifact before acting.
* The metric-fix check finished its four lenses and 13 of its adjudications. Adjudicated items
  are marked; the rest are candidates.

Already known and deliberately not re-listed: figure 12's caption says "peptides" for a figure
that pools peptides and propeptides; the Russian generator `datascale_tol_plot.py` still carries
the panel-(a) mislabel that was fixed in the English figure.


## A. Paper fact-check — 24 candidate findings, none adjudicated

### [major] Adapter's "net +287,558" parameters is wrong: it double-counts an aliased BiLSTM and a disabled boundary head

- **Where:** 212
- **As written:** the projection itself holds 331{,}008 parameters, partly offset because the convolutions downstream now read 256 channels instead of 1280, for a net $+287{,}558$.
- **Problem:** The stated mechanism does not produce the stated number. 331,008 (projector) minus the conv1 saving (122,912 params at 1280 channels -> 24,608 at 256, i.e. 98,304 saved) is 232,704, not 287,558. The extra 54,854 is an artifact of counting state_dict entries rather than parameters: (a) GatedResidualConvProjectedLSTMCNN does `self.biLSTM = self.backbone.biLSTM` (src/models/crf_models.py:1318), so the same 50,176-parameter BiLSTM is serialized under two names and counted twice; (b) the adapter-only run also instantiates a 4,678-parameter boundary head that is switched off in that config (`"boundary_state_scale": 0.0`), so it contributes nothing to the adapter. 50,176 + 4,678 + 232,704 = 287,558 exactly.
- **Correction:** The adapter's own net cost is +232,704 (= 331,008 - 98,304), matching the sentence's own arithmetic. The checkpoint's net over the base, including the inert boundary head, is +237,382 (462,092 - 224,710).
- **Evidence:** torch.load('runs/5cv_esm2_adapter_only/outer0_inner1/model.pt'): state_dict entries sum to 522,671 but data_ptr comparison shows feature_extractor.biLSTM.* alias feature_extractor.backbone.biLSTM.* (8 tensors, 50,176 params); unique total 472,495. Instantiating the model via src.train_loop_crf.get_model on that cell's config.json gives sum(p.numel() for p in m.parameters() if p.requires_grad) = 462,092 for the adapter and 224,710 for the base. The training log logs/5cv_esm2_adapter_only_log_o0_i1.txt itself records "trainable params:  462092". runs/5cv_esm2_adapter_only/outer0_inner1/config.json has "boundary_state_scale": 0.0.

### [major] "an embedding five times wider" -- ESM-C 6B is exactly 2x wider than ESM-2, not 5x

- **Where:** 305
- **As written:** The intervals overlap, so an embedding five times wider buys no confirmed improvement on its own, while either architectural addition on the narrower one does.
- **Problem:** The ESM-C 6B per-residue embedding is 2560-dimensional and the ESM-2 650M one is 1280-dimensional: a factor of 2. No quantity in the pipeline is a factor of 5 -- the parameter ratio 6B/650M is about 9.2x, and the paper's own appendix (line 455) writes "ESM-C 6B compression 2560$\to$256", confirming the 2560 width.
- **Correction:** "an embedding twice as wide" (or, if the point is model capacity, "a pLM roughly nine times larger").
- **Evidence:** runs/5cv_esmc6b_plain/outer0_inner1/config.json has "embedding_dim": 2560; runs/5cv_baseline_esm2/outer0_inner1/config.json has "embedding_dim": 1280. torch.load on data/uniprot_2022/embeddings/embeddings_esmc6b/000b462ed3390aa2530e3faa2309772c.pt gives torch.Size([59, 2560]); the same protein in data/uniprot_2026/embeddings/emb_esm2 gives torch.Size([59, 1280]). Corroborated by the ESM-C model's conv1 shape (32, 2560, 3) in its checkpoint.

### [major] "trainable stack of 235,113" counts 10,403 non-trainable CRF constraint-mask buffers

- **Where:** 210 (repeated in the appendix at line 340)
- **As written:** It costs 4{,}678 parameters on top of a trainable stack of 235{,}113, next to a frozen pLM of 650 million.
- **Problem:** 235,113 is the sum of the base model's state_dict entries, which includes three registered buffers that carry no gradient: crf._constraint_mask (101x101 = 10,201), crf._constraint_start_mask (101) and crf._constraint_end_mask (101), i.e. 10,403 non-trainable values. The repo's own training code prints the correct figure.
- **Correction:** The base model has 224,710 trainable parameters (235,113 - 10,403). Both occurrences (Sec. 2 and the appendix's "giving 235{,}113 trainable parameters for the base model") should say 224,710, or be relabelled as a checkpoint-size figure rather than a trainable-parameter count.
- **Evidence:** Instantiating the base config with src.train_loop_crf.get_model prints "trainable params:  224710"; sum(p.numel() for p in m.parameters() if p.requires_grad) = 224710, sum(b.numel() for b in m.buffers()) = 10403, state_dict sum = 235113. The buffer tensors are visible in runs/5cv_baseline_esm2/outer0_inner1/model.pt as crf._constraint_mask (101,101), crf._constraint_start_mask (101), crf._constraint_end_mask (101). The same +10,403 offset applies to every configuration (adapter logs say 462,092 while its state_dict sums to 522,671).

### [major] The ESM-C 6B row is not evaluated on the same proteins as the ESM-2 rows, contradicting "the same protocol" and "the five folds used here"

- **Where:** 224 and 148 (abstract), 267, 305
- **As written:** 8{,}897 of them carry ESM-2 embeddings and enter the five folds used here.  ...  the base architecture is statistically indistinguishable from its ESM-2 counterpart under the same protocol ($0.588\pm0.016$)
- **Problem:** The four ESM-2 rows of Table 1 use graphpart_assignments_5motif.esm2covered.csv (8,897 proteins, outer-fold test sizes 1558/2572/1263/2025/1479). The ESM-C 6B row uses graphpart_assignments_5motif.esmc6bcovered.csv (8,999 proteins, test sizes 1580/2600/1273/2054/1492). 123 proteins are scored only in the ESM-C run and 21 only in the ESM-2 runs, so the head-to-head 0.588 vs 0.573 is on different test sets (and different training sets). Nothing in the main text flags this; "the five folds used here" is true only of the ESM-2 rows.
- **Correction:** Either state that the ESM-C row is scored on a 8,999-protein embedding-coverage set that differs from the ESM-2 set by 144 proteins, or re-score both on the 8,876-protein intersection before calling the comparison "the same protocol".
- **Evidence:** runs/5cv_esmc6b_plain/outer0_inner1/config.json: "partitioning_file": ".../graphpart_assignments_5motif.esmc6bcovered.csv"; runs/5cv_baseline_esm2/outer0_inner1/config.json: "...esm2covered.csv". wc -l gives 8898 and 9000 lines (8,897 / 8,999 rows). Per-cell tolerance_metrics.json "n_proteins" fields: ESM-2 folds 1558/2572/1263/2025/1479, ESM-C folds 1580/2600/1273/2054/1492. Set comparison of the two CSVs: 8,876 common (identical fold labels), 123 esmc-only, 21 esm2-only. analysis/experiments/nested_cv_queue.jsonl records the two different split files.

### [major] The "intervals overlap" test is applied only to ESM-C; by the same test neither single addition is confirmed either

- **Where:** 305
- **As written:** The intervals overlap, so an embedding five times wider buys no confirmed improvement on its own, while either architectural addition on the narrower one does.
- **Problem:** The sentence rules ESM-C 6B out because its mean +- std interval overlaps the base's, then asserts that the boundary head and the adapter each clear the bar. They do not, under that same criterion: base 0.5725 +- 0.0253 = [0.5473, 0.5978]; boundary head 0.5967 +- 0.0261 = [0.5706, 0.6227]; adapter 0.6018 +- 0.0262 = [0.5756, 0.6280]. Both overlap the base substantially. Only the combined row, 0.6302 +- 0.0210 = [0.6092, 0.6512], is disjoint from the base interval. (A defensible distinction does exist -- the two additions beat the base on 5 of 5 outer folds while ESM-C wins on only 4 of 5 -- but the paper does not report it and invokes interval overlap instead.)
- **Correction:** Either drop the second clause, or replace the overlap test with the paired per-fold evidence: the head is +0.034/+0.019/+0.022/+0.016/+0.030 (5/5 folds), the adapter +0.064/+0.031/+0.023/+0.021/+0.007 (5/5), while ESM-C is +0.016/+0.016/+0.033/+0.017/-0.007 (4/5).
- **Evidence:** runs/5cv_*/nested_cv_tolerance.json, cv_tol3_all_f1_mean/std: base 0.572523/0.025254, boundary 0.596663/0.026084, adapter 0.601807/0.026185, full 0.630193/0.020985, esmc 0.587692/0.015705. Per-fold deltas computed from the per_outer_tol3_all_f1 blocks of the same files.

### [major] "their spread exceeds every effect measured on them" is false under the paper's own definition of spread, and false at the exact-match tolerance under any definition

- **Where:** 187 and 312
- **As written:** homology-aware folds are not exchangeable, their spread exceeds every effect measured on them, and averaging it away costs roughly 200 GPU-hours per configuration.
- **Problem:** Sec. 4.3 defines the reported spread as "the standard deviation across outer folds", which is 0.025 for the base at +-3 -- smaller than the +0.058 combined effect the paper headlines. The claim survives only if "spread" silently means the fold range (0.061), and even then only barely and only at +-3: at an exact match the base's five outer-fold scores are 0.3527/0.3476/0.3476/0.3318/0.3470, a range of 0.021 and a std of 0.008, against a reported combined effect of +0.086.
- **Correction:** Restrict the claim to the +-3 tolerance and name the statistic, e.g. "the range across folds (0.061 at +-3) exceeds each single-addition effect"; it does not hold for the combined configuration at an exact match.
- **Evidence:** main.tex line 238 defines the spread as the std across outer folds. runs/5cv_baseline_esm2/nested_cv_tolerance.json: cv_tol3_all_f1_std = 0.025254, per_outer_tol3_all_f1 range 0.5985-0.5378 = 0.0607; cv_tol0_all_f1_std = 0.007925, per_outer_tol0_all_f1 range 0.3527-0.3318 = 0.0209. Combined-vs-base gaps: +0.0577 at +-3, +0.0862 at exact (runs/5cv_esm2_full/nested_cv_tolerance.json minus baseline).

### [major] Table 5 has no standard deviations, but the sentence that introduces it promises them — and without them the table is a duplicate of Table 2

- **Where:** 482-484 (text) and 488-496 (the table); the included file is texs/ai4dd/figures/tolerance_table.tex
- **As written:** \Cref{tab:tolerance} repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level
standard deviation on every cell.
- **Problem:** The generated table that is \input at line 495 contains only point estimates. In the built PDF (page 14, Table 5) every cell reads "0.573 / 0.540 / 0.461 / 0.345 / +0.000" with no ± anywhere. Since Table 2 already prints the same four F1 columns and the same growth column to the same three decimals, Table 5 as built carries zero information the main table does not, and the only stated reason for its existence is the thing it is missing. This is the appendix's one nested-CV table, and it is the promised home of the per-tolerance spread that the tolerance argument in Section 5 rests on.
- **Correction:** Regenerate figures/tolerance_table.tex with the std that already exists in the CSV: ESM-2 base $0.573\pm0.025$, $0.540\pm0.026$, $0.461\pm0.017$, $0.345\pm0.008$; + boundary head $\pm0.026$, $\pm0.024$, $\pm0.016$, $\pm0.015$; + adapter $\pm0.026$, $\pm0.024$, $\pm0.011$, $\pm0.020$; + both $\pm0.021$, $\pm0.023$, $\pm0.016$, $\pm0.013$; ESM-C 6B base $\pm0.016$, $\pm0.013$, $\pm0.007$, $\pm0.013$. (Or delete Table 5 and the sentence.)
- **Evidence:** texs/ai4dd/figures/tolerance_table.tex — five data rows, no ± token in the file. texs/ai4dd/figures/tolerance_table.csv has the std columns (tol3_std…tol0_std), e.g. "ESM-2, base",0.5725,0.0253,0.5399,0.0255,0.4607,0.0165,0.3453,0.0079. Source of truth: runs/5cv_baseline_esm2/nested_cv_tolerance.json → cv_tol3_all_f1_std = 0.025254, cv_tol2 = 0.025539, cv_tol1 = 0.016494, cv_tol0 = 0.007925 (same keys in the other four runs). Rendered PDF confirmed with pymupdf on main.pdf page 14.

### [major] "235,113 trainable parameters" is the size of the whole state_dict; the trainable count is 224,710

- **Where:** 340 (appendix A, "Baseline architecture: DeepPeptide"); the same number is repeated in the main text at line 292
- **As written:** The configuration we train throughout uses 32 convolutional filters of width 3 and an LSTM hidden size of 64 per direction, giving 235{,}113 trainable parameters for the base model.
- **Problem:** 235,113 is the total numel of every tensor in the checkpoint, which includes the CRF's three constant constraint masks (101x101 + 101 + 101 = 10,403 entries). Those are registered with register_buffer, not nn.Parameter, so they are never optimized and are not parameters at all — they are the 0/1 legality mask for the extended state graph. The repo's own parameter counter, which uses sum(p.numel() for p in model.parameters()), reports 224,710 for exactly this configuration. The number is load-bearing twice over: the appendix uses it to size the base model, and the main text uses it as the denominator for the boundary head's cost ("4,678 parameters on top of a trainable stack of 235,113").
- **Correction:** 235{,}113 -> 224{,}710 trainable parameters (and the same substitution at line 292). The 4,678 figure for the head is correct as written (LayerNorm 128 + Linear 64x64+64 = 4,160 + Linear 64x6+6 = 390).
- **Evidence:** runs/5cv_baseline_esm2/outer0_inner1/model.pt: sum over the state_dict = 235,113; the three _constraint* tensors sum to 10,403; 235,113 - 10,403 = 224,710. src/models/multi_tag_crf.py:74 is `self.register_buffer('_constraint_mask', constraint_mask)` (the nn.Parameter versions on lines 57-58 and 73 are commented out). analysis/metrics/clean_model_params.csv: baseline_esm2,224710, produced by analysis/metrics/src/generators/thesis_eval.py:36 `npar=sum(p.numel() for p in model.parameters())`. Config confirms 32 filters / kernel 3 / hidden 64: runs/5cv_baseline_esm2/outer0_inner1/config.json.

### [major] The structural-projection width sweep is disclaimed as confounded by encoder hidden size, but the three arms differ in nothing except the projection width

- **Where:** 474 (appendix D.3, "Structural-projection width and telescopic CRF")
- **As written:** The width of the structural-feature projection (16/32/48 units) left pooled F1 flat at $0.691$, $0.697$ and $0.693$, although the narrowest arm also differs in encoder hidden size, so this is not a clean width-only sweep;
- **Problem:** The three F1 values are right, but the caveat is not. Diffing the three run configs leaves exactly two keys that differ: out_dir and struct_proj_size (16 / 32 / 48). hidden_size is 64 in all three, num_filters 32, kernel_size 3, seq_proj_size 256, model class lstmcnncrf_gated3di_boundary, same partitioning file, same 2,322 evaluated proteins. The checkpoints agree: biLSTM.weight_hh_l0 is (256, 64) in all three and seq_projector.proj.weight is (256, 2560) in all three; only struct_projector.proj.weight moves, (16,20) -> (32,20) -> (48,20). So this IS a clean width-only sweep, and the paper is throwing away a usable result by disclaiming it.
- **Correction:** Drop "although the narrowest arm also differs in encoder hidden size, so this is not a clean width-only sweep". The sweep is clean; the finding is simply that structural-projection width does not matter over 16-48 units.
- **Evidence:** runs/2026_esmc6b_3di_gated_boundary/config.json, runs/2026_esmc6b_3di_struct32/config.json, runs/2026_esmc6b_3di_struct48/config.json — the only differing keys are out_dir and struct_proj_size {16, 32, 48}. Checkpoint shapes from the three model.pt files as above. F1 values from analysis/metrics/clean_2026_table.csv: esmc6b_3di_gated_boundary 0.69124, esmc6b_3di_struct32 0.69717, esmc6b_3di_struct48 0.69338, all at n_test 2322.

### [major] The LoRA numbers are from the 2022 dataset and a five-fold split, not from "the same protocol" the section defines

- **Where:** 472 (appendix D.3, "LoRA fine-tuning"); the protocol it points back to is defined at line 429
- **As written:** Low-rank adaptation of the last few pLM layers, instead of fully frozen embeddings, scored $0.558$ and $0.567$ against $0.621$ for the frozen baseline under the same protocol.
- **Problem:** Section D.1 defines the protocol as "seven GraphPart folds" on the 2026 rebuild. All three runs behind this sentence are 2022-dataset runs on a five-fold partitioning. The 0.621 baseline is runs/train_run_esm2, whose F1 on the 2026 seven-fold protocol has no relation to 0.621 — the 2026 frozen ESM-2 baseline is 0.5711 pooled (0.5378 on model-select, 0.6056 on sealed). Presenting 0.558/0.567 against 0.621 as a same-protocol comparison imports a baseline that is 0.05 F1 above the one every other number in this section is measured against, which makes the LoRA penalty look larger and better-grounded than the evidence supports.
- **Correction:** Either say the LoRA arms were run on the earlier 2022 build against its own 0.621 frozen baseline (so "under the same protocol" becomes "under an earlier 2022-release protocol"), or drop the numbers and keep only the qualitative statement. The comparison is internally valid (all three share a data file and partitioning); it is only "the same protocol" that is false.
- **Evidence:** runs/esm2_lora_lstmcnncrf/config.json and runs/esm2_lora_lstmcnncrf_r4_last2_qv/config.json: data_file = data/uniprot_2022/labeled_sequences.csv, partitioning_file = data/uniprot_2022/graphpart_assignments.csv. Same two fields in runs/train_run_esm2/config.json. data/uniprot_2022/graphpart_assignments.csv has 5 folds (clusters 0-4, 7623 rows); data/uniprot_2026/graphpart_assignments.csv has 7 (0-6, 8994 rows). Values: analysis/errors/error_stats/type_agnostic_metrics.csv — esm2_lora_lstmcnncrf f1_all 0.55808, esm2_lora_lstmcnncrf_r4_last2_qv 0.56713, train_run_esm2 0.62080. 2026 baseline for contrast: analysis/metrics/clean_2026_table.csv baseline_esm2 f1 0.5711.

### [major] The dual-matcher agreement claim overstates the worst case and the "exactly" is false for one of the five configurations

- **Where:** 419 (appendix C, last paragraph)
- **As written:** The two agree to $0.0016$ F1 at worst and to $3\times10^{-7}$ at the median, and the aggregate reproduces the published summary exactly.
- **Problem:** The median is right (3.15e-7 over the 100 cells on all-segment F1). The other two halves are not. Recomputing the buggy matcher on the re-decoded segments and differencing against each cell's published test_f1_all gives a worst case of 5.22e-4, not 1.6e-3 (5.22e-4 on all-segments, 1.25e-3 if peptide-only F1 is included in the pool) — so the stated bound is roughly 3x looser than the data. And the aggregate does not reproduce exactly: 5cv_esm2_adapter_only's published cv_f1_all_mean is 0.58162820 while the re-scored original-matcher aggregate is 0.58166910, and its per-outer vector differs in the fourth decimal on all five folds (0.6005/0.607/0.5414/0.5701/0.5892 published vs 0.6006/0.6069/0.5415/0.57/0.5893 recomputed). Sharper: five of that run's twenty cells exceed the rescore script's own self-validation gate, SANITY_TOL = 1e-4, which the script's docstring says means "the partition/checkpoint reconstruction for that cell is wrong, and its corrected number should NOT be trusted". The other four configurations agree to ~5e-7 throughout, so the discrepancy is specific to the adapter run and a reader deserves to know it exists.
- **Correction:** "The two agree to $5\times10^{-4}$ F1 at worst and to $3\times10^{-7}$ at the median. Four of the five configurations reproduce the published summary to within $5\times10^{-7}$; the adapter-only run differs by $4\times10^{-5}$ in the aggregate, with five of its twenty cells above the $10^{-4}$ self-validation threshold, so its inference was not bit-identical on re-run."
- **Evidence:** Per-cell diff of runs/*/outer{o}_inner{k}/corrected_metrics.json["orig_all_f1"] against runs/*/outer{o}_inner{k}/cell_result.json["test_f1_all"] over all 100 cells: max 0.000522 (5cv_esm2_adapter_only outer2_inner3), median 3.148e-7; including peptides/propeptides F1 the max is 0.001252 (same cell). Aggregates: runs/5cv_esm2_adapter_only/nested_cv_summary.json cv_f1_all_mean 0.5816281996801966 vs nested_cv_summary_corrected.json cv_orig_all_f1_mean 0.5816690999999999. Gate: analysis/experiments/rescore_nested_cv_corrected.py:103 SANITY_TOL = 1e-4, used at line 199; the five adapter cells above it are (0,2) 3.8e-4, (1,2) 2.1e-4, (2,3) 5.2e-4, (3,0) 3.5e-4, (4,1) 4.7e-4.

### [minor] Abstract's "with little loss from applying them together" reverses the direction: the combination is superadditive

- **Where:** 148
- **As written:** both additions outperform the base architecture on ESM-2 embeddings individually ($+0.024$ and $+0.029$ F1) and combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base, with little loss from applying them together.
- **Problem:** "little loss" states sub-additivity. The measured combination exceeds the sum of the parts: +0.024140 + +0.029284 = +0.053424 against a measured +0.057670, i.e. 108% of the sum. The Results section (line 297) gets this right with "close to the sum of the parts"; the abstract does not.
- **Correction:** "with no loss from applying them together" or "which is slightly more than the sum of the parts".
- **Evidence:** runs/5cv_*/nested_cv_tolerance.json, cv_tol3_all_f1_mean: base 0.572523, boundary 0.596663 (+0.024140), adapter 0.601807 (+0.029284), full 0.630193 (+0.057670).

### [minor] Conclusion's "by less than half that on their own" is false for the adapter

- **Where:** 310
- **As written:** improve segment F1 by $+0.058$ together and by less than half that on their own.
- **Problem:** Half of the combined gain (+0.057670) is +0.028835. The adapter alone gains +0.029284, which is more than half, not less. Only the boundary head (+0.024140) is below half. On the rounded numbers the paper itself prints, the adapter's +0.029 is exactly half of +0.058, so the claim fails on both the rounded and the unrounded figures.
- **Correction:** "and by roughly half that on their own", or state the two separately (+0.024 and +0.029).
- **Evidence:** runs/5cv_esm2_full/nested_cv_tolerance.json cv_tol3_all_f1_mean 0.630193 minus runs/5cv_baseline_esm2 0.572523 = 0.057670; runs/5cv_esm2_adapter_only 0.601807 - 0.572523 = 0.029284 > 0.028835.

### [minor] "nudges boundaries inward" has no supporting artifact and the only signed displacement data points the other way

- **Where:** 303
- **As written:** on ESM-2 the head filters spurious segments and nudges boundaries inward as a side effect, while the adapter both finds more segments and places their ends more precisely.
- **Problem:** "Inward" is a directional claim, but every statistic cited in that paragraph is symmetric and unsigned: boundary_error_cv.py scores each true segment by min over predictions of max(|dstart|,|dend|), and the paired paragraph reports the share of that error equal to zero. The one artifact that does carry signs, segment_quality_cv.json (dstart = pred_start - true_start, dend = pred_end - true_end), shows the boundary head moving starts earlier and ends later than the base -- i.e. outward, lengthening segments -- on both segment types.
- **Correction:** Drop "inward", or replace it with the signed statistic actually measured. Note also that any such directional statement inherits the terminus problem: a predicted boundary that coincides with the chain terminus cannot be displaced outward, so the signed means are only interpretable on interior boundaries.
- **Evidence:** analysis/metrics/src/boundary_error_cv.py, seg_error(): `e = max(abs(ps - ts), abs(pe - te))`, so the reported paired statistic is start/end symmetric. analysis/metrics/segment_quality_cv.json, model-level metrics: peptides_dstart_signed_mean -0.1608 (base) -> -0.4309 (boundary head), peptides_dend_signed_mean +0.0739 -> +0.4314; propeptides_dstart_signed_mean -0.1010 -> -0.3869, propeptides_dend_signed_mean +0.4165 -> +0.5248. Sign convention from analysis/metrics/src/segment_quality_cv.py:202, `dstart[task].append(ps - ts)`.

### [minor] The "segment ends" claim is not what the cited comparison measures

- **Where:** 186 and 310
- **As written:** the head buys precision, the adapter buys recall, and both place segment ends more precisely, measured on the segments a variant and the base model both find.
- **Problem:** The comparison named here ("the segments a variant and the base model both find") is the paired analysis of Sec. 5, whose statistic is the share of paired segments where max(|dstart|,|dend|) = 0 -- both boundaries exactly right. It cannot distinguish a start improvement from an end improvement, so it cannot support a claim specifically about ends. (End-specific evidence does exist in analysis/metrics/segment_quality_cv.json -- paired d_abs_dend of -0.044, -0.068 and -0.085 for head, adapter and both -- but it is not the number the paper reports, and it is not in the paper at all.)
- **Correction:** Either say "place segment boundaries more precisely" (matching the symmetric statistic actually reported), or report the end-specific paired deltas and restrict them to interior boundaries.
- **Evidence:** analysis/metrics/src/boundary_error_cv.py, seg_error() returns max(abs(ps-ts), abs(pe-te)); the reported quantities exact_base/exact_variant in analysis/metrics/boundary_error_cv.json are np.mean(pb == 0) over that symmetric error. The end-specific numbers are in analysis/metrics/segment_quality_cv.json under paired_vs_baseline/tol3/d_abs_dend_mean.

### [minor] Table 1's caption promises a std on every cell; six of the twenty F1 cells have none, and the appendix table it points to has none at all

- **Where:** 271-275 (Table 1 caption); 482-495 (the table it refers the reader to)
- **As written:** $5\times4$ nested cross-validation with the corrected matcher, mean $\pm$ std.\ over the
five outer folds.
- **Problem:** Only the precision, recall and +-3 F1 columns of Table 1 carry a standard deviation; the +-2, +-1 and exact F1 columns are bare means. The main text sends the reader to Table 4 for the fuller version ("repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level standard deviation on every cell"), but the generated file it \input{}s, texs/ai4dd/figures/tolerance_table.tex, has five columns of bare means and no standard deviations anywhere -- although the stds are present in the sibling CSV and in the source JSON.
- **Correction:** Either say "mean +- std over the five outer folds for P, R and F1 at +-3; means for the tightened tolerances", or regenerate tolerance_table.tex with the tol2_std/tol1_std/tol0_std columns that figures/tolerance_table.csv already carries.
- **Evidence:** texs/ai4dd/figures/tolerance_table.tex contains only `ESM-2, base & $0.573$ & $0.540$ & $0.461$ & $0.345$ & $+0.000$`; texs/ai4dd/figures/tolerance_table.csv has the matching tol2_std 0.0255, tol1_std 0.0165, tol0_std 0.0079 columns, and runs/5cv_*/nested_cv_tolerance.json has cv_tol{0,1,2}_all_f1_std for every run.

### [minor] Table 4's 3Di row reports the peptide side of the trade-off as -0.04; the artifact says -0.033

- **Where:** 454 (the "Structural channel (3Di)" row of \Cref{tab:verdict})
- **As written:** Structural channel (3Di) & extra input & propep.\ $+0.02$ / pep.\ $-0.04$, net trade-off & no effect \\
- **Problem:** Recomputing the net-3Di contrast under the corrected matcher (esmc6b_3di_gated_boundary against its 3Di-zeroed control esmc6b_3di_zeroctrl, pooled over the two held-out folds, common protein set n = 2,322) gives peptide dF1 = -0.0335, which rounds to -0.03, not -0.04. The propeptide half is right (+0.0208) and so is the "no effect" verdict (net -0.0022, CI [-0.012, +0.008]). Only the peptide magnitude is inflated by one unit in the second decimal, and it is the number that makes the trade-off look asymmetric.
- **Correction:** propep.\ $+0.02$ / pep.\ $-0.03$, net trade-off
- **Evidence:** analysis/metrics/clean_regime_protfp.csv, per-protein tp/fn/fp for esmc6b_3di_gated_boundary vs esmc6b_3di_zeroctrl on folds {2,5}, split by task: pep dF1 = -0.0335 (bootstrap CI [-0.0512, -0.0159]), propep dF1 = +0.0208 (CI [+0.0094, +0.0325]), all tasks -0.0022 (CI [-0.0123, +0.0075]), n = 2322. Cross-checks: per-fold from analysis/metrics/clean_split_modelselect.csv (pep 0.6940 vs 0.6934 = +0.0006) and clean_split_sealed_test.csv (pep 0.6214 vs 0.6849 = -0.0635) average to -0.031. Neither basis yields -0.04.

### [minor] "a confident $+0.03$" contradicts the same table's own row text and the fold-level numbers three paragraphs earlier

- **Where:** 440 (appendix D.2, first paragraph), against line 451 (the table row) and line 429
- **As written:** nested cross-validation puts ESM-C 6B at $0.588\pm0.016$ against ESM-2's $0.573\pm0.025$ (\Cref{tab:main-results}), overlapping intervals, where this table records a confident $+0.03$.
- **Problem:** The table row this sentence is characterizing reads "$+0.03$, unstable across folds", and Section D.1 has already reported that this exact contrast measured +0.074 on one held-out fold and -0.011 on the other. Calling it "a confident $+0.03$" is the opposite of what the paper says about it in both other places, and it weakens the argument being made: the point is that a pooled interval excluding zero survived a modification whose per-fold estimates flip sign, which is a stronger illustration of the protocol's weakness than a straw "confident" claim.
- **Correction:** "...where this table records $+0.03$ on a pooled interval that excludes zero, despite per-fold estimates of $+0.074$ and $-0.011$."
- **Evidence:** main.tex:451 "$+0.03$, unstable across folds"; main.tex:429 "$+0.074$ on one and $-0.011$ on the other". Both verified: analysis/metrics/clean_split_paired_triage.csv rows "plain pLM: 6B vs ESM2" give 0.0742 (fold 2) and -0.0111 (fold 5); pooled over the common 2,325 proteins from analysis/metrics/interaction_perprotein_2026.csv the estimate is +0.0315, CI [+0.009, +0.052].

### [nit] Three numbers are off by 0.001 from the artifact, all consistent with rounding twice through four decimals

- **Where:** 297 (recall delta), 286 and 292 (Table 1)
- **As written:** The adapter is mostly a recall effect: recall rises by $0.042$ against $0.013$ of precision.  ... $0.616 \pm 0.020$ ... $0.561 \pm 0.027$
- **Problem:** Adapter recall delta: 0.579163 - 0.537700 = 0.041463, which rounds to 0.041, not 0.042. Table 1 base precision: 0.615488 rounds to 0.615, not 0.616. Table 1 ESM-C recall: 0.560485 rounds to 0.560, not 0.561. All three become the printed value only if the artifact value is first rounded to four decimals (0.0415, 0.6155, 0.5605) and then to three, so this looks like one transcription habit rather than three independent slips. Every other cell of Table 1 (37 of 40 numbers) rounds correctly in one step.
- **Correction:** 0.041 (adapter recall gain), 0.615 (base precision), 0.560 (ESM-C recall); or generate the table from the JSON rather than transcribing it, as is already done for figures/tolerance_table.tex.
- **Evidence:** runs/5cv_esm2_adapter_only/nested_cv_tolerance.json cv_tol3_all_recall_mean = 0.579163; runs/5cv_baseline_esm2/nested_cv_tolerance.json cv_tol3_all_recall_mean = 0.537700 and cv_tol3_all_precision_mean = 0.615488; runs/5cv_esmc6b_plain/nested_cv_tolerance.json cv_tol3_all_recall_mean = 0.560485. All three re-derived independently from the 100 per-cell tolerance_metrics.json files (agreement < 5e-6).

### [nit] "Every gap widens as the tolerance tightens" is not monotone for two of the three variants

- **Where:** 301
- **As written:** Every gap widens as the tolerance tightens, from $+0.024$ to $+0.039$ for the boundary head, $+0.029$ to $+0.051$ for the adapter and $+0.058$ to $+0.086$ for the two together
- **Problem:** The endpoints are right, but the gap does not widen monotonically. The adapter's gap goes 0.0293 -> 0.0323 -> 0.0304 -> 0.0509 and the combined model's 0.0577 -> 0.0591 -> 0.0571 -> 0.0862, both dipping between +-2 and +-1. Only the boundary head is monotone (0.0241 -> 0.0263 -> 0.0269 -> 0.0391). The following clause ("each is nearly flat from $\pm3$ to $\pm1$") describes the real shape, so the opening generalisation overstates it.
- **Correction:** "Every gap is wider at an exact match than at +-3" -- which is what the four quoted numbers actually show.
- **Evidence:** analysis/metrics/boundary_error_cv.json abs_gap blocks: boundary {3: 0.0241, 2: 0.0263, 1: 0.0269, 0: 0.0391}; adapter_only {3: 0.0293, 2: 0.0323, 1: 0.0304, 0: 0.0509}; full {3: 0.0577, 2: 0.0591, 1: 0.0571, 0: 0.0862}. Re-derived from runs/5cv_*/nested_cv_tolerance.json.

### [nit] Table 3's median-peptide-length row gives the 2022 column a value the 2022 data does not have

- **Where:** 363 (the last data row of \Cref{tab:dataset})
- **As written:** Median peptide length (residues) & 20--21 & 20--21 \\
- **Problem:** Computed from the two labeled_sequences.csv files: 2022 median peptide length is 21 and median propeptide length is 21; 2026 median peptide is 20 and median propeptide is 21. So "20--21" is a fair summary of the 2026 column (peptides 20, propeptides 21) but not of the 2022 column, where nothing is 20. The row label also says "peptide", so the range is presumably meant to cover both segment types — which is worth saying, since as printed the row's only job is to assert the two releases are identical on this axis, and they are not quite.
- **Correction:** Either give the two numbers the row claims to report — 2022: 21, 2026: 20 — or relabel the row "Median segment length (peptide/propeptide)" and print 21/21 for 2022 and 20/21 for 2026.
- **Evidence:** data/uniprot_2022/labeled_sequences.csv: 6,372 peptide segments, median length 21.0; 8,211 propeptide segments, median 21.0. data/uniprot_2026/labeled_sequences.csv: 7,431 peptide segments, median 20.0; 9,140 propeptide segments, median 21.0. (Every other cell in this table is exact — see confirmations.)

### [nit] The motif-clustering description says four residues per boundary; the code uses four residues per segment

- **Where:** 371-372 (appendix B, second paragraph)
- **As written:** obtained by $k$-means ($k=50$) on ESM-2 embeddings of the four residues
flanking each annotated boundary
- **Problem:** k=50 is right and the four-residue count is right, but they are not four residues per boundary. The clustering script takes, for a segment [s, e], the positions {s-2, s-1, e+1, e+2} — two residues outside the N-terminal boundary and two outside the C-terminal boundary — and concatenates their four embeddings into one 5120-d vector per *segment*, then clusters those. As written a reader reconstructs a two-residues-each-side window around a single cleavage site, which is a different feature and a different number of clustered items.
- **Correction:** "...on ESM-2 embeddings of the two residues flanking each end of an annotated segment, concatenated into one vector per segment"
- **Evidence:** analysis/dataset/src/flanking_motif_clusters.py — docstring: "Flank window for a 1-based segment [s, e]: positions {s-2, s-1, e+1, e+2} ... Concatenation order: [N-2, N-1, C+1, C+2] -> 4*1280 = 5120-d"; flank_vector() line ~65 `pos = [s - 2, s - 1, e + 1, e + 2]`; K = 50 at line 42, KMeans(n_clusters=args.k, random_state=SEED, n_init=10) at line 110.

### [nit] Table 4's gated-adapter row is an ESM-C 6B measurement but, unlike its neighbours, does not say so

- **Where:** 456 (the "Gated adapter" row of \Cref{tab:verdict}); read against line 440
- **As written:** Gated adapter (pLM re-projection) & input adapter & $+0.022$, CI $[+0.008, +0.038]$ & helps \\
- **Problem:** The number is exactly reproducible, but it comes from an ESM-C 6B pair: runs/2026_esmc6b_adapter256_seqonly (embedding_dim 2580, 2560->256 re-projection, 3Di zeroed, seq-only) against runs/2026_esmc6b_boundary. The two rows directly above it are labelled "Boundary head on ESM-C 6B" and "Boundary head on ESM-2" precisely because the embedding matters here, and line 440 then counts this row among the three "later re-tested under nested cross-validation" — where the adapter was tested on ESM-2 only (1280->256). A reader comparing +0.022 to the nested-CV +0.029 will not know the two are on different embeddings.
- **Correction:** Label the row "Gated adapter on ESM-C 6B (pLM re-projection)", to match the two boundary-head rows.
- **Evidence:** runs/2026_esmc6b_adapter256_seqonly/config.json: embedding_dim 2580, model lstmcnncrf_gated3di_boundary, embeddings dir embeddings_esmc6b_3dizero. Paired against runs/2026_esmc6b_boundary (embedding_dim 2560) on the common 2,361 proteins of analysis/metrics/adapter256_perprotein_2026.csv and analysis/metrics/clean_regime_protfp.csv, corrected matcher: dF1 = +0.0222, bootstrap CI [+0.0077, +0.0380] — the table's numbers to three decimals. Per fold: -0.0010 (fold 2), +0.0448 (fold 5), which is also line 429's "$-0.001$ on one fold and $+0.045$ on the other". Method's adapter is 1280->256 on ESM-2 (main.tex:296).

### [nit] "on peptides alone it does not hold at all" is true of the mean but not of every fold

- **Where:** 421 (appendix C, last sentence)
- **As written:** per outer fold the full ordering holds in four folds of five after correction and three of five before it, and on peptides alone it does not hold at all.
- **Problem:** The two fold counts are exactly right. The peptide clause overstates by one fold: the mean ordering on peptides does fail (base 0.5632 < head 0.5891 but adapter 0.5804 < head, so base < head < adapter < both breaks), and it fails on four of the five outer folds — but it does hold on outer fold 1, where the corrected peptide F1s are 0.6312 < 0.6496 < 0.6594 < 0.6859. "Not at all" invites a reader to check and find a counterexample in a paragraph whose whole point is careful accounting of how often the ordering survives.
- **Correction:** "...and on peptides alone the mean ordering does not hold, surviving in only one fold of five."
- **Evidence:** Per-outer means of corr_peptides_f1 over the four inner cells, from runs/5cv_*/outer{o}_inner{k}/corrected_metrics.json — fold0 0.6351/0.6795/0.6707/0.6853 (fails), fold1 0.6312/0.6496/0.6594/0.6859 (holds), fold2 0.5574/0.5472/0.5568/0.5846 (fails), fold3 0.4713/0.5059/0.4928/0.5232 (fails), fold4 0.5209/0.5634/0.5223/0.5900 (fails), in base/head/adapter/both order. Means 0.5632/0.5891/0.5804/0.6138, matching nested_cv_summary_corrected.json cv_corr_peptides_f1_mean in each run.


## B. Metric-suite fix check — 28 findings, 12 adjudicated (11 confirmed)

Concerns `analysis/metrics/src/segment_quality_cv.py` and the numbers derived from it.

### [blocker] F5's interior filter is joint, not per-end: all four quoted "interior-only" displacements are wrong

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:212-219 (`interior = ts != 1 and (L is None or te != L)`), emitting `<task>_dstart_interior_abs_mean`, `<task>_dend_interior_abs_mean`, `<task>_n_matched_interior`
- **Claim:** A matched segment is excluded from BOTH `dstart_in` and `dend_in` if EITHER of its ends is terminal. So `dstart_interior` is not "the displacement of interior starts" - it is the displacement of starts of segments that are interior at both ends, i.e. a start metric conditioned on an end property. The stated rationale ("a boundary that IS the chain terminus cannot be overshot, so its error is half-truncated by construction") justifies dropping that boundary, not the whole segment.
- **Consequence:** All four quoted interior numbers must be replaced. Baseline, 20 cells, same aggregation: peptides code 1.828/1.677 vs correct per-end 2.024/1.793 (raw 2.008/1.419); propeptides code 1.070/2.834 vs correct 1.640/3.480 (raw 1.497/2.570). The propeptide start error is off by 53%. The qualitative start-vs-end narrative survives either convention (peptide gap 0.59 raw -> 0.15 joint / 0.23 per-end; propeptide gap 1.07 -> 1.76 / 1.84), so this is a blocker on the printed values, not on the claim they support. `n_matched_interior` (540.7 for peptides) is mislabeled the same way: the per-end counts are 802.5 starts and 559.7 ends out of 821.5 matched.
- **Evidence:** Strongest single piece: under the code, propeptide |dstart| moves DOWN from raw (1.497 -> 1.070) when "half-truncated" boundaries are removed. The stated mechanism predicts UP, because starts at residue 1 are the artificially easy ones - and the per-end filter does give 1.640. The drop is entirely the co-excluded C-terminal-ending segments. Reproduced on 5cv_esm2_full too: peptides code 1.299/1.101 vs per-end 1.429/1.248; propeptides code 0.811/1.918 vs per-end 1.204/2.389. Independent cross-check (agrees to within a protein-set difference, keep=None vs keep=shared): forcing every length to be missing so only `ts != 1` can fire yields peptides_dstart_interior = 2.0217, against the per-end 2.0235.

### [major] Claim (1) "widens monotonically" is false — pooled and in 4/5 folds

- **Where:** analysis/metrics/segment_quality_cv.json, models.*.metrics.all_f1_iou{0.5..0.95}.per_outer
- **Claim:** The gap does not widen monotonically with the IoU threshold. Pooled over folds it is monotone for +head (+0.0016/+0.0093/+0.0141/+0.0191/+0.0207/+0.0329/+0.0345) and +head+adapter, but NOT for +adapter, which reverses at 0.75->0.8 (+0.0271 -> +0.0254), nor for ESM-C (+0.0106 -> +0.0102). Per outer fold the sequence is monotone in only 1/5 folds for +head (outer3), 1/5 for +adapter (outer1), 2/5 for +head+adapter (outer1, outer3). Outer2 systematically runs the other way for all three of +adapter (+0.0336@0.7 -> +0.0147@0.8), +head+adapter (+0.0678@0.6 -> +0.0493@0.9) and ESM-C (+0.0494@0.6 -> +0.0154@0.9): there the gap NARROWS as the criterion tightens. Restricting to the three thresholds actually quoted (0.5/0.9/0.95) does not rescue it: 0.9->0.95 reverses in outer0 and outer1 for +head (+0.0356->+0.0322, +0.0301->+0.0262), in outer0 for +adapter (+0.0578->+0.0548) and in outer0 for +head+adapter (+0.0757->+0.0677).
- **Consequence:** A reviewer who asks for the per-fold table sees the claim contradicted in the majority of folds, and in one whole fold the direction is reversed for three of the four models. The defensible statement is the endpoint one: the gap at IoU 0.95 exceeds the gap at 0.5 in 5/5 folds (+head), 4/5 (+adapter), 4/5 (+head+adapter). Say "the gap grows as the threshold tightens, from +0.002 to +0.034 for the head" and drop the word "monotonically".
- **Evidence:** Per-fold delta rows computed from all_f1_iou*.per_outer for the 7 thresholds; monotonicity violations enumerated above. Pooled adapter row: +0.0196 +0.0222 +0.0256 +0.0271 +0.0254 +0.0334 +0.0374.

### [major] F5: `<task>_dstart_interior_abs_mean` is not "interior starts"; quoting propeptide 1.070 as an interior-start error is wrong by ~50%

- **Where:** analysis/metrics/src/segment_quality_cv.py:212 (`interior = ts != 1 and (L is None or te != L)`) and :318-323
- **Claim:** The interior flag is a JOINT condition: a segment contributes to dstart_interior only if its start is not residue 1 AND its end is not the C-terminus. For propeptides the two conditions are on disjoint populations (crosstab: ts==1 & te==L co-occur 0 times; 24956 = 15263 + 3203 + 6490), and the te==L subset has mean |dstart| = 2.852 against 0.920 for te<L. So excluding C-terminally-ended segments drags |dstart| DOWN from 1.566 (starts not at residue 1) to 1.020 (both ends interior) — the 1.070 in the JSON. Same effect for peptides: 1.957 -> 1.747. This directly contradicts the stated F5 rationale, which says removing anchored boundaries should RAISE the error because it is half-truncated by construction. That holds for |dend| (propeptides 2.626 raw -> 3.494 on te<L) but the opposite happens to |dstart|, and the cause is the te==L half of the filter, not the ts==1 half.
- **Consequence:** The joint filter is the RIGHT instrument for a start-vs-end contrast (both ends measured on the same segments), so this is not a computation bug and the asymmetry conclusion is unaffected. But the field name and the paper's "interior-only 1.070 / 2.834" invite a reader to extract "interior propeptide starts are placed to ~1 residue". The quantity that answers that question is 1.566. Relabel the field (e.g. `dstart_both_ends_interior_abs_mean`), state in the caption that both numbers are conditioned on the same doubly-interior subset, and fix the code comment, which currently explains only the |dend| direction.
- **Evidence:** Recomputed from the raw dumps over the intersected protein set (baseline, 24956 matched propeptide pairs): |dstart| ts==1 = 0.448 (n=3203), ts>1 = 1.566 (n=21753), te==L = 2.852 (n=6490), te<L = 0.920 (n=18466), ts>1 AND te<L = 1.020 (n=15263, the reported figure). Peptides: ts>1 = 1.957, ts>1 AND te<L = 1.747.

### [major] Claim (3) mis-attributes the paired effect: the adapter ALONE is the strongest single ingredient, not "the two together"

- **Where:** analysis/metrics/segment_quality_cv.json, models.*.paired_vs_baseline.{iou50,tol3}.d_iou_mean
- **Claim:** "The boundary head alone is not distinguishable from zero while the two additions together are" is literally true but implies neither addition alone is distinguishable. +adapter alone is: tol3 +0.0051+-0.0017, t=6.96, p=0.002, 5/5 folds — the tightest and most significant entry in the entire paired table, stronger than +head+adapter (+0.0073+-0.0052, t=3.15, p=0.034). Under iou50 the adapter alone is +0.0097+-0.0053, t=4.12, p=0.015, 5/5. Separately, ESM-C is also 5/5 under tol3 (+0.0041+-0.0031, t=2.97, p=0.041), a positive paired result the summary omits. Finally the null for the head is gate-dependent: clean under tol3 (t=1.02, p=0.367, 3/5) but marginal under the declared PRIMARY_GATE iou50 (t=2.18, p=0.095, 4/5).
- **Consequence:** As written the sentence reads as "the additions only work in combination", which the data contradicts — the adapter is the load-bearing ingredient and is significant on its own. Also, quoting the head's null from tol3 while quoting the combination's positive from iou50 would be gate-shopping across a 4-model x 3-gate table with no multiplicity correction. Name one gate, report the head's number under that same gate, and credit the adapter.
- **Evidence:** t-tests over the 5 outer-fold means, all gates: iou50 head +0.0052+-0.0053 t=2.18 p=0.095 4/5; adapter +0.0097+-0.0053 t=4.12 p=0.015 5/5; full +0.0143+-0.0064 t=5.02 p=0.007 5/5; esmc +0.0046+-0.0041 t=2.52 p=0.065 4/5. tol3 head +0.0020+-0.0043 t=1.02 p=0.367 3/5; adapter t=6.96 p=0.002; full t=3.15 p=0.034; esmc t=2.97 p=0.041.

### [major] Claim (4) quotes the weak statistic: 0.344 vs 0.404 is 4/5 and p=0.08; the robust result is matched-only 0.60 vs 0.75, and it is long PROPEPTIDES

- **Where:** analysis/metrics/segment_quality_cv.json, iou_len45-50 / iou_len45-50_matched_only / match_rate_len45-50
- **Claim:** Not a protein-set artifact — F4 is verified (see confirmations), so the effect is real. But the number chosen is the noisy one. iou_len45-50 (mean over ALL true): 0.344 vs 0.404, delta -0.060+-0.058, t=-2.31, p=0.082, negative in only 4/5 folds (outer2 is +0.018). iou_len45-50_matched_only: 0.602 vs 0.747, delta -0.146+-0.034, t=-9.69, p=0.0006, negative in 5/5 folds. Match rate is statistically indistinguishable (0.545 vs 0.523, delta +0.021+-0.068, t=0.70, p=0.52), so the F1 fix's warning about self-selected subsets does not bite for this particular comparison and matched-only is simply the sharper instrument. Split by task, the 5/5 matched-only result is carried entirely by propeptides (-0.147+-0.100, t=-3.30, 5/5, n=731 matched) — peptides are -0.074+-0.222, t=-0.74, 4/5, n=215, not robust. The bin is 66.6% propeptides (1148 of 1724).
- **Consequence:** As phrased ("long-segment problem", 0.344 vs 0.404, juxtaposed with match rates 0.52/0.54) a reader will take it as a DETECTION deficit, which the data rule out — ESM-C finds at least as many long segments and places them much worse. Under-sold as well as mis-framed: the supportable claim is stronger than the one being made. Write "ESM-C localises long propeptides worse: mean IoU on matched 45-50 segments 0.60 vs 0.75, in all 5 folds, at an indistinguishable match rate", and drop or footnote the 0.344/0.404 pair as the fold-unstable version.
- **Evidence:** Per-fold matched_only deltas: -0.160 -0.131 -0.198 -0.117 -0.123 (5/5). Per-fold iou_len deltas: -0.141 -0.066 +0.018 -0.074 -0.037 (4/5). n_true_len45-50 per outer = [135, 112, 56, 62, 66] per cell, so three folds rest on <70 segments. Per-task matched_only recomputed from the raw dumps over the intersected set.

### [major] Claim (5) propeptide half is under-sold: the raw asymmetry is NOT robust (4/5, p=0.18); only the interior correction establishes it

- **Where:** analysis/metrics/segment_quality_cv.json, 5cv_baseline_esm2 propeptides_dstart/dend_abs_mean vs *_interior_abs_mean
- **Claim:** "The propeptide asymmetry survives" implies it was already established and merely persists. It was not. Raw |dstart|-|dend| = -1.072+-1.472, t=-1.63, p=0.179, negative in only 4/5 folds — outer2 flips sign to +0.711. The doubly-interior version is -1.764+-0.815, t=-4.84, p=0.008, negative in 5/5 folds. Per-end filtering agrees (-1.928, 5/5). So the interior correction does not preserve the effect, it creates a defensible one out of a fold-unstable one.
- **Consequence:** The paper is leaving its strongest evidence on the table and quoting a number a reviewer can knock down by asking for the per-fold spread. Lead with the interior figure and state that the raw contrast is fold-unstable. (Peptide half of the claim holds as stated: raw +0.589, t=5.66, 0/5 negative, collapsing to +0.152, t=0.66, p=0.55 interior — asymmetry gone.)
- **Evidence:** Per-fold baseline |ds|-|de|: propeptides raw -3.289 -1.220 +0.711 -1.219 -0.345; interior -2.956 -2.076 -1.741 -1.163 -0.882. Peptides raw +0.499 +0.335 +0.772 +0.892 +0.450; interior +0.275 -0.698 +0.500 +0.595 +0.087.

### [major] F5 "interior" is a both-ends filter, not the per-end control its stated rationale describes

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:212 — `interior = ts != 1 and (L is None or te != L)`; consumed at lines 306-311 to emit `<task>_dstart_interior_abs_mean` / `<task>_dend_interior_abs_mean`
- **Claim:** The justification given for F5 is per-boundary: "a boundary that IS the chain terminus cannot be overshot, so its error is half-truncated by construction". That argument licenses excluding ts==1 from |dstart| and te==L from |dend|. The code instead excludes a segment from BOTH statistics if EITHER of its ends is terminal, so `dstart_interior` is additionally purged of segments whose C-end is the terminus (an exclusion with no bearing on dstart truncation) and `dend_interior` is purged of segments starting at residue 1.
- **Consequence:** The quoted "interior-only" numbers are not the quantity the paper's sentence describes, and the discrepancy is large for propeptides. Baseline propeptide |dstart|: raw 1.497, correct per-end control (ts!=1) 1.640, emitted 1.070 — 35% below the correct value. Baseline propeptide |dend|: raw 2.570, correct per-end control (te!=L) 3.480, emitted 2.834 — 19% below. Note the emitted "interior" dstart (1.070) sits BELOW the raw value (1.497), whereas removing the half-truncated ts==1 segments empirically raises it to 1.640; the entire drop is produced by the te==L exclusion. Peptides move less but still move: baseline |dend| raw 1.419, te!=L 1.793, emitted 1.677. The start-vs-end contrast the section argues for survives either way (propeptide end still ~2x worse than start), but every quoted magnitude changes.
- **Evidence:** Independent recomputation over all 20 cells x 5 models (my reimplementation reproduces the emitted `*_interior_abs_mean` to <1e-5 against the JSON `per_outer` lists, then adds per-end variants). Per-end corrected table, |dstart| over ts!=1 / |dend| over te!=L, for direct substitution — peptides: baseline 2.024/1.793, boundary 1.781/1.534, adapter 1.728/1.489, full 1.429/1.248, esmc6b 2.572/2.001. Propeptides: baseline 1.640/3.480, boundary 1.417/2.498, adapter 1.249/2.772, full 1.204/2.389, esmc6b 1.631/2.289. Fix: either compute dstart over ts!=1 and dend over te!=L (matches the stated rationale), or keep the both-ends subset and rename the fields to say "both ends interior" — the current name and the paper sentence claim the former, the code does the latter.

### [major] The sign-consistency discipline F2 introduced was not applied to the headline F1@IoU table, and two of the quoted deltas have no supportable sign

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/segment_quality_cv.json — `models.<m>.metrics.all_f1_iou0.5.per_outer`; contrast with the `n_outer_positive` block added at segment_quality_cv.py:506-511
- **Claim:** F2 added `n_outer_positive` to the paired test with the explicit rationale (segment_quality_cv.py:505-507) that "a mean +- std over 5 folds hides how consistent the sign is; a 5/5 sign count is p = 0.031 one-sided where 1.4 sigma reads as noise". The same test is absent from the unpaired F1@IoU deltas that head the quoted results. Applied to them, the F1@0.5 row does not survive: the boundary head is +0.0016 with across-fold SD 0.0143 and only 2 of 5 outer folds positive; ESM-C 6B is +0.0061 with SD 0.0243 and 2 of 5 positive.
- **Consequence:** "+0.002" and "+0.006" at IoU 0.5 will be read as gains by a NeurIPS reviewer. The across-fold spread is 9x and 4x the mean respectively and the sign flips in 3 of 5 folds, so neither is distinguishable from zero. This is exactly the failure mode F2 was written to prevent, left in place one table over. (Not a code defect and not one of F1-F5; a reporting gap the fixes create by raising the standard unevenly.)
- **Evidence:** Recomputed from the JSON `per_outer` lists. all_f1_iou0.5 baseline per-outer mean 0.6660; deltas: boundary +0.0016 (SD 0.0143, 2/5 positive), adapter +0.0196 (SD 0.0162, 5/5), full +0.0365 (SD 0.0152, 5/5), esmc6b +0.0061 (SD 0.0243, 2/5). The 0.9 and 0.95 rows are sound: @0.9 +0.0329 (5/5) / +0.0334 (5/5) / +0.0636 (5/5) / +0.0140 (4/5); @0.95 +0.0345 / +0.0374 / +0.0706 / +0.0153, all 5/5. Recommend adding n_outer_positive to the F1@IoU table, or reporting the two 0.5-threshold cells as null.

### [major] F5's `interior` filter is segment-level, not boundary-level, and moves |dstart| opposite to its own stated rationale

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:212 — `interior = ts != 1 and (L is None or te != L)`, feeding dstart_in/dend_in at :217-219
- **Claim:** The fix is justified per boundary ("a boundary that IS the chain terminus cannot be overshot") but implemented per segment: a segment is dropped from BOTH dstart_interior and dend_interior if EITHER end is terminal. So `<task>_dstart_interior_abs_mean` is computed on a sample selected by the segment's END property. The stated mechanism (removing artificially-easy, half-truncated boundaries) can only RAISE the mean absolute error; the shipped propeptide value falls instead.
- **Consequence:** The two numbers the paper will quote as "interior-only" are for a different estimand than the one described, and the discrepancy is large on propeptides. Per-boundary filtering (|dstart| over all matched segments with ts!=1, |dend| over all with te!=L) gives baseline propeptide 1.640 / 3.480 against the reported 1.070 / 2.834 — the reported |dstart| is 35% low (1.070 vs 1.640, i.e. the true figure is 53% higher). Peptides: 2.024 / 1.793 against the reported 1.828 / 1.677. The qualitative conclusions (peptide start-vs-end gap narrows, propeptide gap widens) survive; the magnitudes do not. Segment-level filtering is defensible if the intent is a PAIRED start-vs-end contrast on identical segments, but then the field names and the F5 rationale must say so.
- **Evidence:** /tmp/audit/f5.py, baseline 5cv_baseline_esm2, all 20 cells, same keep-set and same per-cell → inner → outer aggregation as the script. propeptides: raw 1.497/2.570, segment-level interior (as coded) 1.070/2.834, boundary-level interior (as claimed) 1.640/3.480. peptides: raw 2.008/1.419, as-coded 1.828/1.677, per-boundary 2.024/1.793. 27.1% of annotated propeptides end at the C-terminus and 13.0% start at residue 1, so the as-coded filter discards ~37% of them selected on the end. Direction check: L-te histogram for propeptides is {0:9032, 1:4, 2:0, 3:20} — no off-by-one, the terminus test itself is correct.

### [major] The four paired deltas in the primary-gate row are computed on four different segment populations, and ESM-C 6B's rank is unstable to that choice

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:464-514 `paired_vs_baseline` / GATES['iou50']; JSON models.<model>.paired_vs_baseline.iou50
- **Claim:** The iou50 gate is evaluated separately per variant, so each row of "+0.0052 / +0.0097 / +0.0143 / +0.0046" is a mean over a different set of true segments (1584.5 / 1736.5 / 1720.0 / 1610.7 admitted pairs per cell). Re-running all four comparisons on one common population — the segments where the baseline and ALL five models have an assigned match with IoU>=0.5, 1360.7 per cell — changes the row substantially and is not a uniform shrink.
- **Consequence:** The row cannot be read as a ranking. On the common population the deltas become +0.0044 / +0.0082 / +0.0105 / +0.000106, i.e. ESM-C 6B collapses from +0.0046 (4/5 folds positive) to +0.0001 (3/5), from roughly tied with the boundary head to nothing, while esm2_full drops 27%. Contribution split of each reported delta: common core / variant-specific extras = 71.8%/28.2% (boundary), 65.7%/34.3% (adapter), 57.8%/42.2% (full), 2.1%/97.9% (ESM-C). This does not make ESM-C's delta spurious — the common set conditions on three ESM-2 variants' outcomes and ESM-C is a different backbone, so the 250 extra segments/cell may be exactly where ESM-2 struggles — but the paper must not present the four numbers as comparable, and should say which population each is on.
- **Evidence:** /tmp/audit/q1e.py and /tmp/audit/q1f.py, reproducing the script's iou50 numbers exactly (+0.005203/+0.009698/+0.014281/+0.004646) before the re-population.

### [major] The primary-gate number reads "+0.0052, sharper" for the boundary head, whose unconditioned segment IoU is 0.017 LOWER than the baseline

- **Where:** JSON models.5cv_esm2_boundary.paired_vs_baseline.iou50.d_iou_mean vs metrics.all_mean_iou_over_true; script /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:461 PRIMARY_GATE
- **Claim:** This is not a computation bug — the gate does exactly what it advertises — but the set of "numbers now reported" contains the gated delta and omits the unconditioned one, and the two have opposite signs for the boundary head. all_mean_iou_over_true is 0.5594 for the boundary head against the baseline's 0.5764, and the paired dIoU over EVERY true segment present in both dumps (unmatched scored 0) is -0.0170 +- 0.0208 with 1/5 folds positive. Decomposed: the boundary head matches 340 true segments per cell that the baseline matches and it does not (contributing -0.0790) against 208 it rescues (+0.0552), on top of +0.0069 from the 1729 both-matched segments.
- **Consequence:** A reviewer who reads "+0.0052 +- 0.0053, 4/5 folds" as "the boundary head places boundaries better" is being shown a conditional statement that reverses sign unconditionally. Under the strictest reported gate the same model gives +0.0020 +- 0.0043 with only 3/5 folds positive, i.e. nothing. If the boundary head is claimed as sharper, all_mean_iou_over_true (or the detection/localisation decomposition above) has to be quoted next to it.
- **Evidence:** /tmp/audit/q1.py (Q1b) and /tmp/audit/q1d.py; script's own stdout: IoU/true = 0.5764 baseline vs 0.5594 boundary. The other three variants do not have this problem (unconditioned dIoU +0.0317 / +0.0420 / +0.0173, 5/5, 5/5, 3/5 folds).

### [minor] The residue-MCC line 0.6710/0.6667/0.6791/0.6952/0.6915 is not a ladder — two rungs are noise and one is negative

- **Where:** analysis/metrics/segment_quality_cv.json, models.*.metrics.all_residue_mcc
- **Claim:** Presented as an ordered sequence it reads as a monotone ladder, but +head is BELOW the baseline (0.6667 vs 0.6710, delta -0.0044+-0.0143, t=-0.68, p=0.53, better in only 2/5 folds) and +adapter is not distinguishable (+0.0081+-0.0158, t=1.15, p=0.32, 3/5). Only +head+adapter moves (+0.0242+-0.0056, t=9.64, p=0.001, 5/5); ESM-C is +0.0205+-0.0220, t=2.08, p=0.106, 4/5. Also full > adapter fails in outer0 and full > esmc fails in outer3 and outer4.
- **Consequence:** Any sentence ordering these five numbers as a progression asserts two increments the data do not support and hides one decrement. Report MCC as "only the combined model improves residue-level MCC (+0.024, 5/5 folds); the head alone slightly reduces it".
- **Evidence:** Per-fold MCC deltas vs baseline: head -0.0237 -0.0133 +0.0044 -0.0013 +0.0121; adapter +0.0335 +0.0085 -0.0007 +0.0075 -0.0084; full +0.0211 +0.0254 +0.0295 +0.0161 +0.0287; esmc -0.0085 +0.0226 +0.0058 +0.0437 +0.0388.

### [minor] F2 defect: `n_outer_positive` has an inverted sign convention for the two displacement metrics

- **Where:** analysis/metrics/src/segment_quality_cv.py:503-508
- **Claim:** The loop attaches n_outer_positive to d_iou_mean, d_abs_dstart_mean and d_abs_dend_mean identically, as `sum(x > 0 for x in per_outer)`. For d_iou_mean positive means the variant is better; for the two |displacement| deltas NEGATIVE means the variant is better. So the JSON records `d_abs_dend_mean: mean -0.1144, n_outer_positive 0` for the boundary head, which is a 5/5 IMPROVEMENT recorded as a 0/5 sign count.
- **Consequence:** Self-inverting for anyone reading the JSON directly or writing a downstream table generator that treats n_outer_positive as "folds where the variant won". None of these two metrics appear in the currently quoted numbers, so no published figure is affected today. Either negate the displacement deltas at construction or emit `n_outer_favourable` with a per-metric direction.
- **Evidence:** JSON, 5cv_esm2_full paired_vs_baseline.iou50: d_abs_dstart_mean mean -0.1587 n_outer_positive 0; d_abs_dend_mean mean -0.2120 n_outer_positive 0. Both are unanimous improvements.

### [minor] The +- in every quoted number is a 5-fold sample std, not a standard error — reverses how two headline numbers read

- **Where:** analysis/metrics/src/segment_quality_cv.py:441-446 (`vals.std()`, pandas ddof=1)
- **Claim:** Aggregation reports std over the 5 outer-fold means. With n=5 the SE is std/2.236, so +0.0052+-0.0053 (head, iou50) looks like 1.0 sigma from zero but is t=2.18, and +0.0143+-0.0064 (full, iou50) looks like 2.2 sigma but is t=5.02.
- **Consequence:** A reviewer eyeballing +-  as a confidence interval will both under-read the significant results and over-read the null. Since the paper's argument turns on "not distinguishable from zero" vs "distinguishable", the convention must be stated in the caption, and the sign counts (already emitted) are the more honest summary: 5/5 is p=0.031 one-sided. With 4 variants x 3 gates and no multiplicity correction, avoid the word "significant".
- **Evidence:** aggregate() uses pandas Series.std() (ddof=1) on the 5 per-outer means; verified against manual t-statistics for every quoted paired entry.

### [minor] `iou_len<bin>` is best-available IoU but `match_rate_len<bin>` is one-to-one assigned — the two cannot be reconciled by a reader

- **Where:** analysis/metrics/src/segment_quality_cv.py:231 vs :396-400
- **Claim:** bin_best accumulates `max(iou(true, q) for q in pred)` over ALL predictions (many-to-one, no assignment), while n_matched_len/match_rate_len count greedy one-to-one assignments and iou_len_matched_only averages the assigned IoUs. For bin 45-50 baseline, match_rate x matched_only = 0.523 x 0.747 = 0.391 but iou_len is 0.404; the 0.013 gap is unassigned overlap.
- **Consequence:** If the paper prints mean-IoU-over-all and match rate side by side (as the summary does: 0.404/.../0.344 with match rates 0.52/.../0.54), a reader who tries to reconcile them will fail and may suspect an error. Harmless to the conclusions; state in the caption that iou_len uses best-available overlap while the match rate is one-to-one.
- **Evidence:** Baseline bin 45-50: iou_len 0.4039, matched_only 0.7474, match_rate 0.5235; product 0.3912 != 0.4039.

### [minor] The F5 rationale "a terminal boundary cannot be overshot, so it is placed almost for free" is unverified for peptide N-termini

- **Where:** analysis/metrics/src/segment_quality_cv.py:208-211 (code comment) and the corresponding paper rationale
- **Claim:** The rationale holds in three of the four cases — peptide C-termini |dend| 0.472 vs 1.901, propeptide C-termini 0.158 vs 3.494, propeptide N-termini |dstart| 0.448 vs 1.566 — but not for peptide N-termini, where anchored starts are slightly HARDER: |dstart| 2.103 at ts==1 (n=380) vs 1.957 at ts>1 (n=16050). A start pinned at residue 1 cannot undershoot but can still overshoot arbitrarily far, so the half-truncation argument does not by itself imply a small error.
- **Consequence:** Not a refutation — n=380 against a high-variance quantity, and only 2.3% of peptide starts — and it changes no reported number. But the rationale is stated as a general geometric fact in both the code comment and the intended paper text, and it is not one. Soften to "C-terminal ends are placed almost for free (0.47 vs 1.90); we remove terminal boundaries so the start/end contrast is measured on comparable boundaries".
- **Evidence:** Baseline, matched pairs over the intersected set: peptides |dstart| ts==1 = 2.103 (n=380) vs ts>1 = 1.957 (n=16050); |dend| te==L = 0.472 (n=5236) vs te<L = 1.901 (n=11194).

### [minor] frac_end_at_c_terminus / frac_start_at_residue_1 are conditioned on detection - the sin F1 was fixed to remove

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:325-329; numerator counted only inside the `for i, j, v in matches` loop, denominator `nm = len(ious[task])`
- **Claim:** Both fractions are computed over MATCHED segments only, so they measure "of the segments this model found, what fraction end at the C-terminus", not the annotation property the fix's rationale appeals to. Each model therefore reports a different denominator population.
- **Consequence:** The quoted "~29% of annotated peptide ends are the C-terminus" is not what the JSON contains; the unconditional rate over all annotations is 27.3%. The source docstring states 32%, which matches neither. Minor because the number is used only as motivation, but it is a model-dependent statistic being presented as a dataset fact.
- **Evidence:** Emitted peptides_frac_end_at_c_terminus = 0.2856 (matched-only) vs 0.2730 over all 61,244 true segments; propeptides 0.2782 vs 0.2712; peptides_frac_start_at_residue_1 = 0.0224 vs 0.0274, propeptides 0.1244 vs 0.1300. Direct histogram of L-te over all true segments: 17,064 of 61,244 have L-te == 0 (27.9% pooled).

### [minor] iou_len<bin> and its companion match_rate/matched_only use two different matching conventions in the same row

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:231 (`bin_best[b] += max(iou over ALL preds)`) vs :207 and :407-411 (`bin_iou` / `match_rate` from the one-to-one assignment)
- **Claim:** `iou_len<bin>` scores each true segment by the best IoU over every prediction, while `iou_len<bin>_matched_only` and `match_rate_len<bin>` come from the greedy one-to-one assignment. An unmatched-but-overlapping segment contributes its overlap IoU to the first and nothing to the other two, so the product identity a reader will assume (matched_only x match_rate ~ iou_len) does not hold, and a merged prediction spanning two true segments credits both.
- **Consequence:** Reported bin 45-50 means are inflated ~0.010-0.015 relative to the one-to-one convention. Ranking and the paper's conclusion are unaffected, but the three numbers printed side by side are not mutually derivable, and a reviewer who checks 0.7474 x 0.5235 = 0.391 against the reported 0.404 will ask why.
- **Evidence:** bin 45-50, best-over-all-preds / assigned-or-0, five models in order: 0.4039/0.3934, 0.3954/0.3849, 0.4477/0.4372, 0.4543/0.4425, 0.3440/0.3289. bin 20-24: 0.7366/0.7119 ... 0.7396/0.7166. bin 5-9: 0.4932/0.4735 ... 0.5796/0.5593. Gap is uniform across models, so no ranking flips.

### [minor] F3 makes all_residue_mcc - the quoted MCC row - count every residue as a true negative twice

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:355-356 and :381-388 (`combined[f"res_{k}"] += res[task][k]`, then `all_residue_mcc`)
- **Claim:** The combined view sums the per-task residue confusion matrices, so a residue that is neither peptide nor propeptide enters as TN once for the peptides task and once for the propeptides task. Pre-existing, but F3 raised the per-cell TN from 382k to 693k, so the double-counted mass now dominates the matrix.
- **Consequence:** The quoted 0.6710 / 0.6667 / 0.6791 / 0.6952 / 0.6915 is a one-vs-rest micro-pooled multi-label MCC over 768,957 residue-slots per cell against 384,479 actual residues. That is a defensible statistic, but it must be labeled as pooled over both segment types rather than printed as "residue MCC"; per-task values differ substantially (peptides 0.6208, propeptides 0.7042 for the baseline).
- **Evidence:** peptides_residue_n_total = propeptides_residue_n_total = 384,478.6 per cell, each equal to the full residue count of the 1,775.2 shared proteins; the combined matrix therefore spans 2x that. F3 itself verified working: TN-on-empty ON vs OFF gives all-MCC 0.6710 vs 0.6525 (+0.0185, matching the code's "~0.019"), peptides 0.6208 vs 0.5484; tp/fp/fn are untouched so residue precision/recall/F1/Jaccard are mathematically unchanged.

### [minor] Missing protein lengths degrade three metrics silently, with no coverage check

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:180 (`L = lengths.get(name)`), :184-191, :212, :263-265, :527-529
- **Claim:** `L is None` is treated as "not the C-terminus" at line 212, so terminal ends silently enter the interior set; the `frac_end_at_c_terminus` numerator undercounts while its denominator does not; and the protein is skipped entirely for TN in both the empty and non-empty branches. main() prints how many lengths it loaded but never checks that against the protein names in the dumps.
- **Consequence:** Does not fire on the current data - zero of the 8,999 protein names lack a length - so no quoted number is affected. It is a silent-failure mode: a rerun against a re-exported or subset labeled_sequences.csv would shift the interior displacements and MCC with no warning. A one-line assertion on coverage closes it.
- **Evidence:** Simulated by dropping lengths at random (seed 0), baseline, keep=None. 20% missing: peptides_dend_interior 1.674 -> 1.574, frac_end_at_c_terminus 0.286 -> 0.234, all_residue_mcc 0.6714 -> 0.6655, n_proteins_with_length 1779 -> 1423. 100% missing: all_residue_mcc = -0.2987, frac_end_at_c_terminus = 0.0000, and the script exits 0 with no diagnostic.

### [minor] Two numbers in the source docstrings are wrong and are the kind that get quoted

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:392-399 ("matches 37 of 87 true segments") and :449-454 ("for the boundary head, 78% of its loose-gate delta")
- **Claim:** `n_true_len<bin>` = 86.2 and `n_matched_len<bin>` = 36.8/47.3 are per-cell MEANS over 20 cells, not counts; the docstring presents them as counts. And the 78% attribution is to the wrong model.
- **Consequence:** If "37 of 87" is lifted into the paper as a sample size it understates the bin by 20x (1,724 true segments in bin 45-50 across the 20 cells). The 78% figure, if attributed to the boundary head, is wrong by 8 points and understates the argument for the strict gates.
- **Evidence:** Recomputed share of the loose-gate dIoU sum coming from pairs where either model has err>3: esm2_boundary 86% (n=34,578 loose pairs, sum +424.48 of which +363.78), esm2_adapter_only 80%, esm2_full 78%, esmc6b_plain 35%. 78% is esm2_full's number, not the boundary head's.

### [minor] F1's `iou_len<bin>` credits best-overlap over all predictions, not the assigned match — so it double-counts and disagrees with `all_mean_iou_over_true`

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:230 — `bin_best[b] += max((iou((ts, te), q) for q in pred), default=0.0)`
- **Claim:** F1 is described two ways that are not the same: "the mean best-available IoU over EVERY true segment" and "an undetected segment scores 0". The code implements the first. Because the max runs over every prediction of the task rather than the one-to-one assignment, a true segment whose only overlapping prediction was assigned to a neighbour still scores, and one prediction can be credited in full to two different true segments.
- **Consequence:** "An undetected segment scores 0" is false for 16.2% of undetected segments, and those get substantial credit (mean 0.525). The JSON therefore carries two incompatible "mean IoU over every true segment" numbers: `all_mean_iou_over_true` = 0.5764 (one-to-one) vs the n_true-weighted mean of the `iou_len` bins = 0.6052 (best-overlap), a +0.029 gap for the baseline, +0.025 to +0.031 across models. A bin curve plotted next to the pooled figure will not reconcile. Between-model deltas are essentially unaffected, so no conclusion moves.
- **Evidence:** Baseline: 19874 of 61392 true segments unmatched, 3222 of them (16.2%) with a nonzero best-overlap IoU, mean 0.525; esm2_full 3107 of 18914 (16.4%), mean 0.563. Bin 45-50, best-overlap vs strict one-to-one per model: 0.4039/0.3934, 0.3954/0.3849, 0.4477/0.4372, 0.4543/0.4425, 0.3440/0.3289 — full-minus-baseline is +0.0504 either way vs +0.0491 strict. Largest gap over all bins is 0.050 (bin 30-34). Fix: reword to "a segment with no overlapping prediction scores 0" and name the field for best-overlap, or switch to the assigned-match convention so the bins agree with `all_mean_iou_over_true`.

### [minor] F5's terminal-anchor fractions are computed over matched segments, so a field describing the annotation varies by model

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:313-317 — `nm = len(ious[task])`, then `frac_start_at_residue_1` and `frac_end_at_c_terminus` divide by `nm`; counters incremented only inside the `for i, j, v in matches` loop at 204-210
- **Claim:** F5 justifies the interior variants with "~29% of annotated peptide ends are the C-terminus" (docstring at line 209 says 32%). The emitted field divides the terminal count by the number of MATCHED segments, not by the number of annotated segments, so it reports each model's self-selected detected subset rather than a property of the data.
- **Consequence:** The number moves with the model — 0.2744 to 0.2934 for peptides across the five runs — when the underlying dataset value is a constant. Quoting any of them as an annotation statistic misstates it, and quoting one model's value in a sentence about the corpus is not reproducible from another model's column. The correct annotated figure is 27.30% for peptide C-termini and 27.12% for propeptide C-termini (2.74% / 13.00% for starts at residue 1). Neither the prompt's ~29% nor the docstring's 32% is right.
- **Evidence:** Recomputed both denominators over all 20 cells. peptides frac_end_at_c_terminus, matched-only (= JSON) vs all-true: baseline 0.2856/0.2730, boundary 0.2883/0.2730, adapter 0.2744/0.2730, full 0.2779/0.2730, esmc6b 0.2934/0.2730. Propeptides: 0.2782, 0.2949, 0.2604, 0.2664, 0.2632 vs a constant 0.2712. The all-true column is identical across models, as it must be.

### [minor] `n_outer_positive` has the opposite sense on the displacement deltas than on d_iou, with no flag

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:508-511 — the same `n_outer_positive` is attached to `d_iou_mean`, `d_abs_dstart_mean` and `d_abs_dend_mean`
- **Claim:** For `d_iou_mean`, positive means the variant is better. For `d_abs_dstart_mean` / `d_abs_dend_mean` (a difference of absolute displacements) positive means the variant is WORSE. The field name, the JSON, and the code comment ("a 5/5 sign count is p = 0.031") give no indication that the reading flips.
- **Consequence:** A reader or script applying the comment's rule uniformly inverts every displacement conclusion: a perfect, all-five-folds improvement is recorded as 0/5 and reads as a failure. Since F2 exists precisely so these counts get quoted, this is a live risk for the table.
- **Evidence:** JSON, gate iou50: `5cv_esm2_full` d_abs_dend_mean = -0.2120 with n_outer_positive = 0 of 5 — the variant's ends are tighter in all five outer folds. Same pattern for adapter (-0.1483, 0/5) and boundary (-0.1144, 0/5). Meanwhile d_iou_mean for full is +0.0143 with 5/5, meaning the opposite thing. Fix: emit `n_outer_better` instead, or negate the displacement deltas so positive is uniformly "better".

### [minor] `n_true_len<bin>` and the other count fields in the JSON are per-cell means, and the code comment already misreads one as a population count

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:439-448 (`aggregate` averages every numeric column) applied to the counts emitted at 405-410; misreading at line 402
- **Claim:** F1 emits `n_true_len<bin>`, `n_matched_len<bin>`, `residue_n_total`, `n_true`, `n_pred`, `n_paired` through the same `aggregate()` that means over inner cells and then over outer folds. The JSON value is therefore a per-cell average carrying a count-like name — `n_true_len45-50` is 86.20, not 86 segments.
- **Consequence:** The file's own comment at line 402 already reads it as a count: "in the 45-50 bin the boundary head matches 37 of 87 true segments against the baseline's 48". Anything copied from there into the paper states a per-cell mean as a dataset total. Because the four inner cells of an outer fold share a test partition, the instance count and the unique count differ by 4x, so there is no single rescaling a reader can apply.
- **Evidence:** 45-50 bin over the 20 scored cells: 1732 segment-instances, 433 unique (outer, protein, span) triples, JSON value 86.20 = 1732/20. `residue_n_total` = 384478.6 is likewise the mean of the five outer test sets' residue totals (cell sizes 1557/2560/1262/2020/1477 proteins), not a corpus figure.

### [minor] The GATES design rationale cites a 78% figure that no aggregation of the shipped data reproduces, and its own diagnostic does not separate iou50 from the loose gate it rejects

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:449-461, the comment block above GATES
- **Claim:** Two checkable statements in the comment fail. (a) "for the boundary head, 78% of its loose-gate delta comes from pairs where one of the two models is outside +-3" — recomputing that share on the current data gives 87.9% under the script's own per-cell → inner → outer aggregation and 85.7% micro-pooled; no combination of keep / match-requirement / pooling reaches 78%, so the number is stale. (b) "Only the strict gates isolate localisation" — applying the same diagnostic to iou50 gives 67.9% / 57.1% / 58.5% / 31.4%, so by the comment's own test iou50 fails too.
- **Consequence:** The comment is what the paper's methods text will paraphrase, and a reviewer who repeats the diagnostic will get different numbers and a conclusion the comment does not support. Note the diagnostic itself is the wrong disqualifier rather than iou50 being wrong: the iou50-admitted pairs that fail tol3 are long (mean true length 25.6 vs 20.3 for tol3-admitted, 55% >=25 residues vs 35%) and in 47.6% of them BOTH models are outside +-3 (base-only 29.9%, variant-only 22.5%), so that mass is long-segment localisation, not the detection gain the headline F1 counts. The fix should either update the numbers and drop the claim, or state the real reason iou50 beats overlap — that overlap admits 53.5+35.4 near-miss detection flips per cell with deltas of +-0.5.
- **Evidence:** /tmp/audit/q2exact.py, /tmp/audit/q2d78.py, /tmp/audit/q2char.py, /tmp/audit/q1.py

### [minor] `frac_end_at_c_terminus` is conditioned on detection and is model-dependent, but is quoted as a property of the annotation

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:326-329 (`anchored` is accumulated inside the matched-pairs loop at :213-216; denominator `nm = len(ious[task])`)
- **Claim:** The numerator counts only matched true segments and the denominator is the matched count, so the field measures "share of the segments this model found whose end is the C-terminus", not "share of annotated peptide ends that are the C-terminus". F5's supporting claim ("~29% of annotated peptide ends are the C-terminus") quotes the matched-conditioned 0.286; the code comment at :209 says 32%; the annotation itself gives 27.3%.
- **Consequence:** A statistic used to justify a methodological choice is itself model-dependent — 0.286 for the baseline, 0.279 for esm2_full on the identical annotation — so it cannot be cited as a fixed property of the data. Magnitude is small (0.286 vs 0.273), but the 32% in the code comment is wrong against both.
- **Evidence:** /tmp/audit/f5.py: peptides frac end==C-term over ALL true = 0.273 (identical for every model, as it must be), over MATCHED = 0.286 (baseline) / 0.279 (esm2_full); JSON peptides_frac_end_at_c_terminus mean = 0.286.

### [minor] The quoted residue MCC is `all_residue_mcc`, which sums the two per-task confusion matrices and so counts every residue twice

- **Where:** /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:355-356 and :381-388
- **Claim:** The values 0.6710 / 0.6667 / 0.6791 / 0.6952 / 0.6915 are `all_residue_mcc`. Each task contributes its own full confusion matrix over the same residues (residue_n_total = 383691 for peptides and 383691 for propeptides in cell (0,1), each exactly equal to the sum of protein lengths), so the pooled matrix covers 767382 residue-slots for 383691 residues.
- **Consequence:** Defensible as a micro-averaged two-label MCC, but it is not "residue-level MCC" in the usual single-label sense and a reviewer recomputing from a single-label confusion matrix will not match. The per-task values differ substantially (peptides 0.6208, propeptides 0.7042 for the baseline), so the pooled figure is not a stand-in for either. Label it, or quote the per-task pair.
- **Evidence:** /tmp/audit/f3.py and /tmp/audit/q3_extra.py; F3's TN arithmetic itself is exactly right (residue_n_total == sum(L), zero overshoot, all 1557 proteins in the cell carry a length).


### Adjudicator verdicts recorded before the stop

- real=False: The factual observation is correct but it is a naming nit, not a defect, and it changes nothing that is quoted.

What I verified. A fresh run reproduces the committed JSON byte-for-byte (`env/bin/python analysis/metrics/src/segment_quality_cv.py --out /tmp/sq_check.json`; `diff` against analysis/metrics/segment_quality_cv.json is empty), so the file under audit is the file that produced the number

- real=True: REAL — confirmed by reading the code and by recomputation over all 20 cells.

Structural confirmation (/home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py):
- Lines 213-216 increment `anchored[task]["start_at_1"]` / `anchored[task]["end_at_terminus"]` INSIDE `for i, j, v in matches:` (the greedy one-to-one IoU match loop, line 197), so only true segments the model matched are e

- real=True: CONFIRMED — verified at the code level and reproduced numerically; the committed JSON is byte-identical to a fresh re-run, so this is the current state.

CODE PATH (/home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py)
- Numerators: lines 214-216 (`if ts == 1: anchored[task]["start_at_1"] += 1`, `if L is not None and te == L: anchored[task]["end_at_terminus"] += 1`) sit INSIDE t

- real=True: CONFIRMED — the code does what the claim says, and every number the claim offers reproduces exactly.

1. Code fact. /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:212 computes a single per-SEGMENT mask, `interior = ts != 1 and (L is None or te != L)`, and lines 217-219 gate BOTH accumulators on it (`dstart_in` and `dend_in`). Lines 319-324 then emit `<task>_dstart_interior

- real=True: CONFIRMED, and the code contradicts its own documentation of fix F1.

CODE FACT (verified by reading /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py):
- Line 231: `bin_best[b] += max((iou((ts, te), q) for q in pred), default=0.0)` — best IoU over EVERY prediction, no assignment.
- Line 224/207: `j, v = matched_iou.get(i, (None, 0.0))` and `bin_iou[(task, b)].append(v)` — th

- real=True: CONFIRMED, and the smoking gun is exact.

1. The mechanism is present as described. `aggregate()` (actual lines 432-446, not 439-448) does `df.groupby("outer").mean(numeric_only=True)` and then means/stds every resulting column with no exemption for counts. The F1 counts emitted at lines 406-411 therefore reach the JSON as a mean over 4 inner cells and then over 5 outer folds. 30 count-named field

- real=True: CONFIRMED, by code reading, exact numerical reproduction, and an independent isolation test.

1. CODE. /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py:212-219 builds ONE predicate and appends to BOTH lists under it:
     interior = ts != 1 and (L is None or te != L)
     if interior:
         dstart_in[task].append(ps - ts)
         dend_in[task].append(pe - te)
   So a mat

- real=True: CONFIRMED. I reran the current script (`env/bin/python analysis/metrics/src/segment_quality_cv.py --out /tmp/sq_repro.json`, 7s) and it reproduces the committed JSON byte-for-byte (md5 2ca28214219b7ce1782a4a6cc54cf253 for both), so the numbers audited are the current ones.

The claim's arithmetic reproduces exactly from `models.<m>.metrics.all_f1_iou0.5.per_outer` (baseline per-outer mean 0.666005

- real=True: CONFIRMED by inspection and by running the code. Line 231 of /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py is `bin_best[b] += max((iou((ts, te), q) for q in pred), default=0.0)` — the assigned-match IoU `v`, computed one line earlier at line 224 from `matched_iou`, is available and is NOT used; the max runs over every prediction of the task.

Method: reproduced the commit

- real=True: CONFIRMED as stated, including its own hedge that it does not fire on today's data.

Code (all four mechanisms verified by reading /home/oskar/work/DeepPeptide/analysis/metrics/src/segment_quality_cv.py):
- :180 `L = lengths.get(name)` returns None silently for an unknown protein.
- :212 `interior = ts != 1 and (L is None or te != L)` — with L None a C-terminal end is classified as interior, so te

- real=True: CONFIRMED — I reproduced every number the claim offered and found stronger evidence than the claim itself put forward.

MECHANISM (code reading, lines 184-191, 263-265, 355-356, 388). `res[task]["tn"]` is accumulated per task; F3 added the empty-pair branch (`res[task]["tn"] += L`) at 184-191; line 355-356 sums tp/fp/fn/tn of BOTH tasks into `combined`; line 388 feeds that combined matrix to `mcc(

- real=True: CONFIRMED, reproduced end-to-end.

1. Code: line 212 is `interior = ts != 1 and (L is None or te != L)` and the single boolean gates BOTH appends at :217-219. A segment is excluded from `dstart_interior` because of its END, and from `dend_interior` because of its START. The stated F5 rationale (code comment :208-211, "A boundary that IS the chain terminus cannot be overshot") is a per-BOUNDARY arg

