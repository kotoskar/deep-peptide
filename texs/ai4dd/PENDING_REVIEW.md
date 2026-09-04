# Review findings — `texs/ai4dd/main.tex`, 2026-09-04

**Nothing here has been applied.** The paper text is untouched.

Status of the review: the 24 findings about NUMBERS were each independently re-derived from
the artifacts by a second agent — 23 confirmed, 1 dismissed. The four other lenses
(consistency, claim strength, bibliography, proofreading) have run but their findings are
**candidates only**: no refuter has tried to kill them. Verify before acting.

---

## 0. Two blockers, verified by hand

### 0.1 The baseline is the only run trained in bf16; every other run is fp32

```
runs/5cv_baseline_esm2      amp=True  amp_dtype=bf16   — all 20 cells
runs/5cv_esm2_boundary      amp=False                  — all 20 cells
runs/5cv_esm2_adapter_only  amp=False                  — all 20 cells
runs/5cv_esm2_full          amp=False                  — all 20 cells
runs/5cv_esmc6b_plain       amp=False                  — all 20 cells
```

Every effect the paper reports is measured against that baseline, so **every effect is
confounded with a change of training precision**: the boundary head's +0.024, the adapter's
+0.029, the combined +0.058 and ESM-C 6B's +0.015. The confound is not hypothetical for this
codebase — an earlier investigation traced a train/infer divergence to exactly this AMP/fp32
difference and the pipeline was switched to forced fp32 afterwards; the baseline run predates
that switch.

This is the paper's central table. It needs either a re-run of the baseline in fp32, or an
explicit statement of the difference and an argument bounding its size. Decide before
submitting; it is the kind of thing a reviewer who reads the released configs will find.

### 0.2 A placeholder reference is printed in the submitted PDF

`refs.bib:283-288`, entry `bpfun2025`, has `journal = {VERIFY-JOURNAL}`. It renders on
**page 8 of `main.pdf`** as `tion. VERIFY-JOURNAL, 2025.` The entry is cited twice, at
`main.tex:196` and `main.tex:338`. The author surnames (`Yang, S. and Shen, H. and Zhang, L.`)
were flagged as possibly invented and have not been checked against the real paper.


---

## 1. Numbers — 23 CONFIRMED by independent recomputation

Each was re-derived from the artifact by a second agent that was told to refute it. The **Evidence** line is that agent's own reproduction, not the original claim.

### [major] "an embedding five times wider" -- ESM-C 6B is exactly 2x wider than ESM-2, not 5x

- **Where:** 305
- **As written:** The intervals overlap, so an embedding five times wider buys no confirmed improvement on its own, while either architectural addition on the narrower one does.
- **Problem:** The ESM-C 6B per-residue embedding is 2560-dimensional and the ESM-2 650M one is 1280-dimensional: a factor of 2. No quantity in the pipeline is a factor of 5 -- the parameter ratio 6B/650M is about 9.2x, and the paper's own appendix (line 455) writes "ESM-C 6B compression 2560$\to$256", confirming the 2560 width.
- **Correction:** "an embedding twice as wide" (2560 vs 1280), or, if the intent was model capacity, "a pLM roughly nine times larger by parameter count".
- **Evidence:** Checked embedding_dim in both configs: runs/5cv_esmc6b_plain/outer0_inner1/config.json has embedding_dim=2560, runs/5cv_baseline_esm2/outer0_inner1/config.json has embedding_dim=1280 (grep confirmed). Ratio is exactly 2x, not 5x. No other pipeline quantity gives 5x either: parameter-count ratio 6B/650M is about 9.2x; the appendix table itself (line 455, verified by grep) writes 'ESM-C 6B compression 2560$\to$256', independently corroborating the 2560 width. There is no reading of the pipeline that produces a factor of 5.

### [major] The "intervals overlap" test is applied only to ESM-C; by the same test neither single addition is confirmed either

- **Where:** 305
- **As written:** The intervals overlap, so an embedding five times wider buys no confirmed improvement on its own, while either architectural addition on the narrower one does.
- **Problem:** The sentence rules ESM-C 6B out because its mean +- std interval overlaps the base's, then asserts that the boundary head and the adapter each clear the bar. They do not, under that same criterion: base 0.5725 +- 0.0253 = [0.5473, 0.5978]; boundary head 0.5967 +- 0.0261 = [0.5706, 0.6227]; adapter 0.6018 +- 0.0262 = [0.5756, 0.6280]. Both overlap the base substantially. Only the combined row, 0.6302 +- 0.0210 = [0.6092, 0.6512], is disjoint from the base interval. (A defensible distinction does exist -- the two additions beat the base on 5 of 5 outer folds while ESM-C wins on only 4 of 5 -- but the paper does not report it and invokes interval overlap instead.)
- **Correction:** Either drop 'while either architectural addition on the narrower one does', or replace the overlap test with a criterion under which the individual additions actually clear the bar (e.g. per-fold win counts: boundary head and adapter each beat the base on 5/5 outer folds vs ESM-C's 4/5), since by mean+-std overlap only the combined configuration is distinguishable from the base.
- **Evidence:** Computed mean+-std intervals at tol3 directly from each run's nested_cv_tolerance.json (cv_tol3_all_f1_mean/std): base [0.5473,0.5978], boundary [0.5706,0.6227], adapter [0.5756,0.6280], full [0.6092,0.6512], esmc [0.5720,0.6034] -- these match the finding's cited numbers almost exactly. Programmatic overlap check against the base interval: boundary overlaps=True, adapter overlaps=True, full overlaps=False, esmc overlaps=True. So by the exact criterion the sentence just used to disqualify ESM-C ('the intervals overlap, so ... buys no confirmed improvement'), the boundary head and the adapter individually also overlap the base and would be equally disqualified; only the combined ('full') configuration clears that bar. The sentence's claim that 'either architectural addition ... does [buy a confirmed improvement]' is inconsistent with the interval-overlap test it just applied to ESM-C.

### [major] "their spread exceeds every effect measured on them" is false under the paper's own definition of spread, and false at the exact-match tolerance under any definition

- **Where:** 187 and 312
- **As written:** homology-aware folds are not exchangeable, their spread exceeds every effect measured on them, and averaging it away costs roughly 200 GPU-hours per configuration.
- **Problem:** Sec. 4.3 defines the reported spread as "the standard deviation across outer folds", which is 0.025 for the base at +-3 -- smaller than the +0.058 combined effect the paper headlines. The claim survives only if "spread" silently means the fold range (0.061), and even then only barely and only at +-3: at an exact match the base's five outer-fold scores are 0.3527/0.3476/0.3476/0.3318/0.3470, a range of 0.021 and a std of 0.008, against a reported combined effect of +0.086.
- **Correction:** Qualify the claim: under the paper's own std-based definition of spread, only the boundary head's individual effect is smaller than the base's fold-to-fold std (0.025 vs 0.024); the adapter (0.029) and the combined effect (0.058) both exceed it. The 'spread exceeds every effect' statement holds only under the informal fold-range reading (0.061) and only at the +-3 tolerance -- at an exact match neither std (0.008) nor range (0.021) exceeds any of the three effects (0.039/0.051/0.086).
- **Evidence:** Confirmed Sec. 4.3 ('Nested cross-validation and its cost', main.tex line 235-238) defines 'the reported spread' as 'the standard deviation across outer folds'. Computed from runs/5cv_baseline_esm2/nested_cv_tolerance.json at tol3: base std=0.025254 (per-outer values 0.5585/0.5943/0.5378/0.5735/0.5985, range=0.0607). Effects vs base: boundary +0.024140, adapter +0.029284, full(combined) +0.057670. Under the paper's own std-based definition, 0.025254 exceeds only the boundary-head effect (0.0241) and is SMALLER than both the adapter's effect (0.0293) and the combined/headline effect (0.0577) -- so 'their spread exceeds every effect measured on them' is false for 2 of the paper's own 3 reported effects under its own stated statistic. The claim only becomes true if 'spread' silently means the informal fold range (0.0607, as used in the adjacent sentence about the boundary head specifically), and even that reading collapses completely at the exact-match tolerance: base std=0.007925 (per-outer 0.3527/0.3476/0.3476/0.3318/0.3470, range=0.0209), against effects of +0.039112 (boundary), +0.050852 (adapter), +0.086235 (full) at tol0 -- neither std nor range exceeds any of the three effects there. This is a repeated claim (contributions bullet, line 187, and conclusion, line 312), not a one-off aside.

### [major] "nudges boundaries inward" has no supporting artifact and the only signed displacement data points the other way

- **Where:** 303
- **As written:** on ESM-2 the head filters spurious segments and nudges boundaries inward as a side effect, while the adapter both finds more segments and places their ends more precisely.
- **Problem:** "Inward" is a directional claim, but every statistic cited in that paragraph is symmetric and unsigned: boundary_error_cv.py scores each true segment by min over predictions of max(|dstart|,|dend|), and the paired paragraph reports the share of that error equal to zero. The one artifact that does carry signs, segment_quality_cv.json (dstart = pred_start - true_start, dend = pred_end - true_end), shows the boundary head moving starts earlier and ends later than the base -- i.e. outward, lengthening segments -- on both segment types.
- **Correction:** on ESM-2 the head filters spurious segments as a side effect (the segments it does place move outward, not inward: peptides_dstart_signed_mean -0.161->-0.431, peptides_dend_signed_mean +0.074->+0.431), while the adapter both finds more segments and localizes them more precisely
- **Evidence:** Read main.tex:303 ("the head ... nudges boundaries inward as a side effect"). Confirmed analysis/metrics/src/boundary_error_cv.py:seg_error() is symmetric/unsigned: e = max(abs(ps-ts), abs(pe-te)), so nothing in that paragraph's cited artifact (boundary_error_cv.json, whose exact_base/exact_variant numbers 0.622->0.653 etc. are the ones quoted in that same paragraph) carries directional information. Checked the one artifact that does carry signs, analysis/metrics/segment_quality_cv.json, with sign convention confirmed at analysis/metrics/src/segment_quality_cv.py:223-224 (dstart = ps-ts, dend = pe-te, so negative dstart = predicted start earlier/N-terminal of true, positive dend = predicted end later/C-terminal of true -- both signs mean the segment got LONGER, i.e. boundaries moved OUTWARD). Read the actual values for peptides: base dstart_signed_mean -0.1608 -> boundary head -0.4309 (more negative = further outward); base dend_signed_mean +0.0739 -> boundary head +0.4314 (more positive = further outward). Same outward pattern for propeptides (dstart -0.1010->-0.3869, dend +0.4165->+0.5248). This is the opposite of 'inward' on both boundaries and both segment types -- confirmed the discrepancy myself, not just on the finding's say-so.

### [major] Table 1's caption promises a std on every cell; six of the twenty F1 cells have none, and the appendix table it points to has none at all

- **Where:** 271-275 (Table 1 caption); 482-495 (the table it refers the reader to)
- **As written:** $5\times4$ nested cross-validation with the corrected matcher, mean $\pm$ std.\ over the
five outer folds.
- **Problem:** Only the precision, recall and +-3 F1 columns of Table 1 carry a standard deviation; the +-2, +-1 and exact F1 columns are bare means. The main text sends the reader to Table 4 for the fuller version ("repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level standard deviation on every cell"), but the generated file it \input{}s, texs/ai4dd/figures/tolerance_table.tex, has five columns of bare means and no standard deviations anywhere -- although the stds are present in the sibling CSV and in the source JSON.
- **Correction:** Table 1 caption: 'mean +- std over the five outer folds for P, R and F1 at +-3; means only for the tightened tolerances.' Table 4 text (line 482) and tolerance_table.tex: either drop the 'standard deviation on every cell' claim or regenerate tolerance_table.tex from tolerance_table.csv to include the tol{2,1,0}_std columns it already computes.
- **Evidence:** Read the Table 1 caption (main.tex:271-273, 'mean +- std. over the five outer folds') against the table body (main.tex:286-292): only P, R and the +-3 F1 column carry a +- std; the +-2, +-1 and exact F1 columns are bare numbers (e.g. '$0.540$ & $0.461$ & $0.345$'), confirmed by inspection of the six cells. Read main.tex:482 ('\Cref{tab:tolerance} repeats the tolerance sweep ... with the fold-level standard deviation on every cell') and then cat'd the actual \input{}'d file texs/ai4dd/figures/tolerance_table.tex: it is `\begin{tabular}{lccccc}` with five bare-number columns and zero +- anywhere -- the text's explicit promise is false for the whole table, not just some cells. Confirmed the source data exists and was simply dropped: figures/tolerance_table.csv has tol2_std=0.0255, tol1_std=0.0165, tol0_std=0.0079 for the ESM-2 base row (matching the finding's cited values exactly) that never made it into the .tex file consumed by \input.

### [major] Table 5 has no standard deviations, but the sentence that introduces it promises them — and without them the table is a duplicate of Table 2

- **Where:** 482-484 (text) and 488-496 (the table); the included file is texs/ai4dd/figures/tolerance_table.tex
- **As written:** \Cref{tab:tolerance} repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level
standard deviation on every cell.
- **Problem:** The generated table that is \input at line 495 contains only point estimates. In the built PDF (page 14, Table 5) every cell reads "0.573 / 0.540 / 0.461 / 0.345 / +0.000" with no ± anywhere. Since Table 2 already prints the same four F1 columns and the same growth column to the same three decimals, Table 5 as built carries zero information the main table does not, and the only stated reason for its existence is the thing it is missing. This is the appendix's one nested-CV table, and it is the promised home of the per-tolerance spread that the tolerance argument in Section 5 rests on.
- **Correction:** Either regenerate figures/tolerance_table.tex to include the ± std already present in the CSV on every cell (ESM-2 base 0.573±0.025 / 0.540±00.026 / 0.461±0.017 / 0.345±0.008; +boundary ±0.026/±0.024/±0.016/±0.015; +adapter ±0.026/±0.024/±0.011/±0.020; +both ±0.021/±0.023/±0.016/±0.013; ESM-C6B base ±0.016/±0.013/±0.007/±0.013), or drop the sentence's promise of per-cell std and delete the now-redundant table.
- **Evidence:** Read main.tex:482-484 ('\Cref{tab:tolerance} repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level standard deviation on every cell') and the \input target texs/ai4dd/figures/tolerance_table.tex verbatim via cat: five rows, each cell a bare point estimate ('$0.573$', '$0.540$', ...), zero ± tokens in the file -- confirms the built PDF table 5 has no std. Diffed against analysis/metrics/figures/tolerance_table.tex (identical, no diff) and its sibling .csv, which does carry tol{3,2,1,0}_std columns (e.g. ESM-2 base: 0.0253/0.0255/0.0165/0.0079) -- so the std values exist and were simply not written into the .tex. Read the generator, analysis/metrics/src/generators/fig_tolerance_cv.py:144-146: the .tex-writing loop only emits `mean`, with a comment 'Compact form for the main text: the fold-level spread is already in tab:main-results and repeating it here only costs width' -- i.e. omitting std was a deliberate choice baked into the code, contradicting the prose that was written to accompany it. Also checked Table 2 (main.tex:280-292) itself: only the ±3 column carries ±std; the ±2/±1/exact columns there are also bare point estimates -- so Table 5, even if read as 'same as Table 2', still doesn't deliver the promised per-tolerance std anywhere, on any of the two tables.

### [major] "235,113 trainable parameters" is the size of the whole state_dict; the trainable count is 224,710

- **Where:** 340 (appendix A, "Baseline architecture: DeepPeptide"); the same number is repeated in the main text at line 292
- **As written:** The configuration we train throughout uses 32 convolutional filters of width 3 and an LSTM hidden size of 64 per direction, giving 235{,}113 trainable parameters for the base model.
- **Problem:** 235,113 is the total numel of every tensor in the checkpoint, which includes the CRF's three constant constraint masks (101x101 + 101 + 101 = 10,403 entries). Those are registered with register_buffer, not nn.Parameter, so they are never optimized and are not parameters at all — they are the 0/1 legality mask for the extended state graph. The repo's own parameter counter, which uses sum(p.numel() for p in model.parameters()), reports 224,710 for exactly this configuration. The number is load-bearing twice over: the appendix uses it to size the base model, and the main text uses it as the denominator for the boundary head's cost ("4,678 parameters on top of a trainable stack of 235,113").
- **Correction:** Replace 235{,}113 with 224{,}710 (trainable parameters) at both occurrences: main.tex line 210 ('...a trainable stack of 235{,}113...') and line 340 (appendix A, '...giving 235{,}113 trainable parameters for the base model.') -- not line 292, which is where the finding mistakenly pointed for the main-text occurrence.
- **Evidence:** Loaded runs/5cv_baseline_esm2/outer0_inner1/model.pt with torch.load and summed v.numel() over the whole state_dict: 235113, matching the paper's figure exactly. Grepped src/models/multi_tag_crf.py: the three constraint tensors (_constraint_start_mask, _constraint_end_mask, _constraint_mask, summing to 101+101+101*101=10403 entries) are registered with `self.register_buffer(...)` at lines 59-60 and 74; the nn.Parameter versions immediately above (lines 57-58, 73) are commented out -- so these 10403 entries are non-trainable buffers, not parameters. 235113-10403=224710. Cross-checked analysis/metrics/clean_model_params.csv (produced by analysis/metrics/src/generators/thesis_eval.py:36, `sum(p.numel() for p in model.parameters())`, which by construction excludes buffers): baseline_esm2,224710 -- exact match. Grepped main.tex for '235': the number appears at line 210 (main text, boundary-head paragraph: 'a trainable stack of 235,113') and line 340 (appendix A) -- NOT at line 292 as the finding states; 292 is a different table row in this build of the file. So the finding's substance is fully correct but its cited main-text line number is wrong.

### [major] The LoRA numbers are from the 2022 dataset and a five-fold split, not from "the same protocol" the section defines

- **Where:** 472 (appendix D.3, "LoRA fine-tuning"); the protocol it points back to is defined at line 429
- **As written:** Low-rank adaptation of the last few pLM layers, instead of fully frozen embeddings, scored $0.558$ and $0.567$ against $0.621$ for the frozen baseline under the same protocol.
- **Problem:** Section D.1 defines the protocol as "seven GraphPart folds" on the 2026 rebuild. All three runs behind this sentence are 2022-dataset runs on a five-fold partitioning. The 0.621 baseline is runs/train_run_esm2, whose F1 on the 2026 seven-fold protocol has no relation to 0.621 — the 2026 frozen ESM-2 baseline is 0.5711 pooled (0.5378 on model-select, 0.6056 on sealed). Presenting 0.558/0.567 against 0.621 as a same-protocol comparison imports a baseline that is 0.05 F1 above the one every other number in this section is measured against, which makes the LoRA penalty look larger and better-grounded than the evidence supports.
- **Correction:** Replace 'under the same protocol' with an explicit note that these three runs are 2022-release, 5-fold-GraphPart runs, not the 7-fold 2026 protocol defined in D.1 -- e.g. 'scored $0.558$ and $0.567$ against $0.621$ for the frozen baseline, all three measured on the earlier 2022 dataset build under its own 5-fold GraphPart split (not the 2026 seven-fold protocol used elsewhere in this appendix).' The internal comparison (LoRA vs. frozen, same data/split) remains valid; only the 'same protocol' language is wrong.
- **Evidence:** Read main.tex:429-430 (appendix D.1, 'Protocol'): defines the whole appendix's single-split baseline as 'seven GraphPart folds ... as in \Cref{sec:dataset}', which section 224 ties to the 2026 rebuild ('8,897 of them carry ESM-2 embeddings and enter the five folds used here' -- actually five outer nested-CV folds, but the underlying GraphPart partitioning for the appendix's single-split protocol is the 7-cluster 2026 file). Checked the three configs behind the LoRA sentence (main.tex:472): runs/esm2_lora_lstmcnncrf/config.json, runs/esm2_lora_lstmcnncrf_r4_last2_qv/config.json, runs/train_run_esm2/config.json all have data_file=data/uniprot_2022/labeled_sequences.csv and partitioning_file=data/uniprot_2022/graphpart_assignments.csv. Loaded both graphpart files with pandas: data/uniprot_2022/graphpart_assignments.csv has 7623 rows and cluster values {0,1,2,3,4} (5 folds); data/uniprot_2026/graphpart_assignments.csv has 8994 rows and cluster values {0..6} (7 folds) -- confirming the LoRA runs used the older, different-fold-count 2022 partitioning, not the '7 GraphPart folds' the appendix defines as its protocol, and nowhere in the appendix text is this distinction disclosed. Verified the cited F1 values in analysis/errors/error_stats/type_agnostic_metrics.csv: esm2_lora_lstmcnncrf f1=0.5581 (~0.558), esm2_lora_lstmcnncrf_r4_last2_qv f1=0.5671 (~0.567), train_run_esm2 f1=0.6208 (~0.621) -- exact matches to the paper's numbers, and the 2026-protocol frozen baseline (analysis/metrics/clean_2026_table.csv, baseline_esm2) is f1=0.5711, ~0.05 lower than the 0.621 the LoRA numbers are compared against.

### [minor] Adapter's "net +287,558" parameters is wrong: it double-counts an aliased BiLSTM and a disabled boundary head

- **Where:** 212
- **As written:** the projection itself holds 331{,}008 parameters, partly offset because the convolutions downstream now read 256 channels instead of 1280, for a net $+287{,}558$.
- **Problem:** The stated mechanism does not produce the stated number. 331,008 (projector) minus the conv1 saving (122,912 params at 1280 channels -> 24,608 at 256, i.e. 98,304 saved) is 232,704, not 287,558. The extra 54,854 is an artifact of counting state_dict entries rather than parameters: (a) GatedResidualConvProjectedLSTMCNN does `self.biLSTM = self.backbone.biLSTM` (src/models/crf_models.py:1318), so the same 50,176-parameter BiLSTM is serialized under two names and counted twice; (b) the adapter-only run also instantiates a 4,678-parameter boundary head that is switched off in that config (`"boundary_state_scale": 0.0`), so it contributes nothing to the adapter. 50,176 + 4,678 + 232,704 = 287,558 exactly.
- **Correction:** The adapter's own net cost by the sentence's stated arithmetic is +232,704 (331,008-98,304), not +287,558. The full checkpoint's net over the base (including the inert 4,678-parameter boundary head) is +237,382 (462,092 trainable vs 224,710).
- **Evidence:** Instantiated both models via src.train_loop_crf.get_model from their actual config.json files (env/bin/python /tmp/check_params.py). Baseline (LSTMCNNCRF): feature_extractor=214,112 (conv1=122,912 @1280ch, biLSTM=50,176, conv2=41,024). Adapter-only (LSTMCNNCRFGated3DiBoundary, config has boundary_state_scale=0.0): projector=331,008, backbone=115,808 (conv1=24,608 @256ch, biLSTM=50,176, conv2=41,024), plus a separate boundary_to_state child = 4,678 params (confirms the paper's own '4,678' figure) that's instantiated but zero-scaled. Confirmed via source read that GatedResidualConvProjectedLSTMCNN does `self.biLSTM = self.backbone.biLSTM` (src/models/crf_models.py:1321, one line off the finding's cited 1318 but the exact same statement) -- an alias, so state_dict serializes the same 50,176 LSTM twice (verified fe.biLSTM is fe.backbone.biLSTM -> True) while model.parameters() correctly dedups it. state_dict sums: adapter 522,671 vs true unique-parameter total 462,092 (matches logs/5cv_esm2_adapter_only_log_o0_i1.txt 'trainable params: 462092' exactly). Doing the sentence's own arithmetic (331,008 projector - (122,912-24,608)=98,304 conv1 saving) gives 232,704, not 287,558; 287,558-232,704=54,854=50,176(aliased biLSTM)+4,678(inert boundary head), exactly as the finding claims.

### [minor] "trainable stack of 235,113" counts 10,403 non-trainable CRF constraint-mask buffers

- **Where:** 210 (repeated in the appendix at line 340)
- **As written:** It costs 4{,}678 parameters on top of a trainable stack of 235{,}113, next to a frozen pLM of 650 million.
- **Problem:** 235,113 is the sum of the base model's state_dict entries, which includes three registered buffers that carry no gradient: crf._constraint_mask (101x101 = 10,201), crf._constraint_start_mask (101) and crf._constraint_end_mask (101), i.e. 10,403 non-trainable values. The repo's own training code prints the correct figure.
- **Correction:** 224,710 trainable parameters, not 235,113. 235,113 is the state_dict total, which additionally includes 10,403 non-trainable CRF buffer values (_constraint_mask, _constraint_start_mask, _constraint_end_mask) that never receive gradients.
- **Evidence:** Same instantiation script: baseline model's true trainable count (sum of p.numel() for p in model.parameters() if p.requires_grad) is 224,710, matching the model's own printed 'trainable params: 224710'. state_dict sum is 235,113 = 224,710 trainable + 10,403 buffers. Read src/models/multi_tag_crf.py: transitions/start_transitions/end_transitions are real nn.Parameters (10,403 total, genuinely trainable), while _constraint_mask (101x101), _constraint_start_mask (101), _constraint_end_mask (101) are registered via self.register_buffer(...) (lines 59-60, 74) -- non-trainable by construction, coincidentally the same total size (10,403) as the real transition parameters, which is what makes 235,113 look plausible as a 'trainable' count when it's actually state_dict-sum. Confirmed both occurrences in main.tex: line 210 ('trainable stack of 235,113') and line 340 in the appendix ('giving 235,113 trainable parameters for the base model').

### [minor] The ESM-C 6B row is not evaluated on the same proteins as the ESM-2 rows, contradicting "the same protocol" and "the five folds used here"

- **Where:** 224 and 148 (abstract), 267, 305
- **As written:** 8{,}897 of them carry ESM-2 embeddings and enter the five folds used here.  ...  the base architecture is statistically indistinguishable from its ESM-2 counterpart under the same protocol ($0.588\pm0.016$)
- **Problem:** The four ESM-2 rows of Table 1 use graphpart_assignments_5motif.esm2covered.csv (8,897 proteins, outer-fold test sizes 1558/2572/1263/2025/1479). The ESM-C 6B row uses graphpart_assignments_5motif.esmc6bcovered.csv (8,999 proteins, test sizes 1580/2600/1273/2054/1492). 123 proteins are scored only in the ESM-C run and 21 only in the ESM-2 runs, so the head-to-head 0.588 vs 0.573 is on different test sets (and different training sets). Nothing in the main text flags this; "the five folds used here" is true only of the ESM-2 rows.
- **Correction:** State explicitly that the ESM-C 6B row is scored on an 8,999-protein embedding-coverage set (144-protein symmetric difference from the ESM-2 8,897-protein set, 8,876 in common with identical fold labels) rather than leaving 'the same protocol' / 'the five folds used here' unqualified for that row.
- **Evidence:** grep'd partitioning_file across all five run configs: the four ESM-2 rows (baseline, boundary, adapter_only, full) all use data/uniprot_2026/graphpart_assignments_5motif.esm2covered.csv; the ESM-C 6B row uses graphpart_assignments_5motif.esmc6bcovered.csv. wc -l gives 8898/9000 lines = 8,897/8,999 proteins, matching the finding. Set comparison in python: 8,876 common, 123 esmc-only, 21 esm2-only, and for every one of the 8,876 common proteins the fold/cluster label is identical between the two files (0 mismatches) -- so it's the same underlying GraphPart clustering, just filtered to a slightly different embedding-coverage subset, not a re-run partitioning. tolerance_metrics.json n_proteins per outer fold: ESM-2 1558/2572/1263/2025/1479 vs ESM-C 1580/2600/1273/2054/1492 for every one of the 5 outer folds -- confirmed by direct read of both runs' JSON files, exactly matching the finding's cited numbers.

### [minor] Abstract's "with little loss from applying them together" reverses the direction: the combination is superadditive

- **Where:** 148
- **As written:** both additions outperform the base architecture on ESM-2 embeddings individually ($+0.024$ and $+0.029$ F1) and combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base, with little loss from applying them together.
- **Problem:** "little loss" states sub-additivity. The measured combination exceeds the sum of the parts: +0.024140 + +0.029284 = +0.053424 against a measured +0.057670, i.e. 108% of the sum. The Results section (line 297) gets this right with "close to the sum of the parts"; the abstract does not.
- **Correction:** and combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base, slightly more than the sum of the two individual gains
- **Evidence:** Read main.tex:148 ("combine to a +0.058 F1 gain ... with little loss from applying them together") and cross-checked cv_tol3_all_f1_mean in runs/5cv_baseline_esm2, 5cv_esm2_boundary, 5cv_esm2_adapter_only, 5cv_esm2_full/nested_cv_tolerance.json. Recomputed: base=0.572523, boundary delta=+0.024140, adapter delta=+0.029284, sum=0.053424, full delta=+0.057670. full-sum=+0.004246 (full is 108.0% of the sum), i.e. the combination is slightly SUPER-additive, not sub-additive. 'little loss' asserts the opposite direction from what the numbers show; line 297 in the same paper ('close to the sum of the parts') is the accurate framing. Confirmed the finding's own numbers reproduce exactly.

### [minor] Conclusion's "by less than half that on their own" is false for the adapter

- **Where:** 310
- **As written:** improve segment F1 by $+0.058$ together and by less than half that on their own.
- **Problem:** Half of the combined gain (+0.057670) is +0.028835. The adapter alone gains +0.029284, which is more than half, not less. Only the boundary head (+0.024140) is below half. On the rounded numbers the paper itself prints, the adapter's +0.029 is exactly half of +0.058, so the claim fails on both the rounded and the unrounded figures.
- **Correction:** improve segment F1 by $+0.058$ together, and individually by $+0.024$ (boundary head) and $+0.029$ (adapter) -- each close to half that
- **Evidence:** Read main.tex:310 ("improve segment F1 by +0.058 together and by less than half that on their own"). Half of the combined tol3 F1 gain (0.057670) is 0.028835. Adapter-only delta = 0.029284 > 0.028835, so the adapter's individual gain is NOT less than half the combined gain -- it's slightly more. Only the boundary head (0.024140) is below half. Even on the paper's own rounded figures (+0.024, +0.029, +0.058), half of 0.058 is 0.029, which equals (not exceeds) the printed adapter gain, so 'less than half' fails even at that precision. Verified by direct computation from nested_cv_tolerance.json.

### [minor] The "segment ends" claim is not what the cited comparison measures

- **Where:** 186 and 310
- **As written:** the head buys precision, the adapter buys recall, and both place segment ends more precisely, measured on the segments a variant and the base model both find.
- **Problem:** The comparison named here ("the segments a variant and the base model both find") is the paired analysis of Sec. 5, whose statistic is the share of paired segments where max(|dstart|,|dend|) = 0 -- both boundaries exactly right. It cannot distinguish a start improvement from an end improvement, so it cannot support a claim specifically about ends. (End-specific evidence does exist in analysis/metrics/segment_quality_cv.json -- paired d_abs_dend of -0.044, -0.068 and -0.085 for head, adapter and both -- but it is not the number the paper reports, and it is not in the paper at all.)
- **Correction:** both place segment boundaries more precisely (matching the symmetric max(|dstart|,|dend|) statistic actually reported), or report the end-specific paired deltas (d_abs_dend_mean: -0.044 head, -0.068 adapter, -0.085 both) restricted to interior boundaries
- **Evidence:** Read main.tex:186 and :310, both say the paired comparison shows the additions 'place segment ends more precisely.' Confirmed the paired comparison actually cited in the text (the 0.608->0.667, 0.622->0.653, 0.611->0.697 'exact' figures at line 303, sourced from analysis/metrics/boundary_error_cv.json's paired.exact_base/exact_variant) is built on seg_error()=max(abs(ps-ts),abs(pe-te)), a symmetric statistic that cannot attribute improvement to the end specifically vs. the start. Checked analysis/metrics/segment_quality_cv.json for the paired_vs_baseline/tol3 block, which does carry separate d_abs_dstart_mean and d_abs_dend_mean: boundary head d_abs_dend_mean=-0.0444 (vs d_abs_dstart_mean=+0.0110, i.e. starts got WORSE on average), adapter d_abs_dend_mean=-0.0677 (vs dstart -0.0184), full d_abs_dend_mean=-0.0847 (vs dstart -0.0280). So an end-specific claim is directionally true in this separate artifact but is not what the cited symmetric comparison measures, and this end/start-split number never appears in the paper. Confirmed both the symmetry of the cited statistic and the existence of the un-cited end-specific numbers myself.

### [minor] The structural-projection width sweep is disclaimed as confounded by encoder hidden size, but the three arms differ in nothing except the projection width

- **Where:** 474 (appendix D.3, "Structural-projection width and telescopic CRF")
- **As written:** The width of the structural-feature projection (16/32/48 units) left pooled F1 flat at $0.691$, $0.697$ and $0.693$, although the narrowest arm also differs in encoder hidden size, so this is not a clean width-only sweep;
- **Problem:** The three F1 values are right, but the caveat is not. Diffing the three run configs leaves exactly two keys that differ: out_dir and struct_proj_size (16 / 32 / 48). hidden_size is 64 in all three, num_filters 32, kernel_size 3, seq_proj_size 256, model class lstmcnncrf_gated3di_boundary, same partitioning file, same 2,322 evaluated proteins. The checkpoints agree: biLSTM.weight_hh_l0 is (256, 64) in all three and seq_projector.proj.weight is (256, 2560) in all three; only struct_projector.proj.weight moves, (16,20) -> (32,20) -> (48,20). So this IS a clean width-only sweep, and the paper is throwing away a usable result by disclaiming it.
- **Correction:** Delete 'although the narrowest arm also differs in encoder hidden size, so this is not a clean width-only sweep' from appendix D.3 (main.tex:474). The three runs are identical except struct_proj_size (16/32/48); the sweep is clean and the finding (flat F1 across widths) can be stated without qualification.
- **Evidence:** Diffed the three config.json files (runs/2026_esmc6b_3di_gated_boundary, .../struct32, .../struct48) key by key in Python: the only differing keys across all three are out_dir and struct_proj_size (16/32/48); hidden_size=64, num_filters=32, kernel_size=3, seq_proj_size=256, model='lstmcnncrf_gated3di_boundary', and partitioning_file are identical in all three. Loaded all three model.pt state_dicts and compared tensor shapes directly: feature_extractor.backbone.biLSTM.weight_hh_l0 is torch.Size([256,64]) in all three (same encoder hidden size), feature_extractor.projector.seq_projector.proj.weight is [256,2560] in all three, and only struct_projector.proj.weight varies, [16,20]->[32,20]->[48,20] -- confirming the checkpoints themselves, not just the configs, differ only in structural-projection width. Also confirmed (per advisor prompt) that the cited F1 values trace to exactly these three runs: grep on analysis/metrics/clean_2026_table.csv gives esmc6b_3di_gated_boundary 0.6912/esmc6b_3di_struct32 0.6972/esmc6b_3di_struct48 0.6934, all at n_test=2322, matching the paper's 0.691/0.697/0.693 and the run names in the config diff. So the paper's caveat ('although the narrowest arm also differs in encoder hidden size') is factually false; this is a clean width-only sweep, and the caveat wrongly disclaims a valid null result.

### [minor] The dual-matcher agreement claim overstates the worst case and the "exactly" is false for one of the five configurations

- **Where:** 419 (appendix C, last paragraph)
- **As written:** The two agree to $0.0016$ F1 at worst and to $3\times10^{-7}$ at the median, and the aggregate reproduces the published summary exactly.
- **Problem:** The median is right (3.15e-7 over the 100 cells on all-segment F1). The other two halves are not. Recomputing the buggy matcher on the re-decoded segments and differencing against each cell's published test_f1_all gives a worst case of 5.22e-4, not 1.6e-3 (5.22e-4 on all-segments, 1.25e-3 if peptide-only F1 is included in the pool) — so the stated bound is roughly 3x looser than the data. And the aggregate does not reproduce exactly: 5cv_esm2_adapter_only's published cv_f1_all_mean is 0.58162820 while the re-scored original-matcher aggregate is 0.58166910, and its per-outer vector differs in the fourth decimal on all five folds (0.6005/0.607/0.5414/0.5701/0.5892 published vs 0.6006/0.6069/0.5415/0.57/0.5893 recomputed). Sharper: five of that run's twenty cells exceed the rescore script's own self-validation gate, SANITY_TOL = 1e-4, which the script's docstring says means "the partition/checkpoint reconstruction for that cell is wrong, and its corrected number should NOT be trusted". The other four configurations agree to ~5e-7 throughout, so the discrepancy is specific to the adapter run and a reader deserves to know it exists.
- **Correction:** "The two agree to $5\times10^{-4}$ F1 at worst (5cv_esm2_adapter_only, outer2/inner3) and to $3\times10^{-7}$ at the median. Four of the five configurations reproduce the published aggregate to within $5\times10^{-8}$; the adapter-only aggregate differs by $4\times10^{-5}$, with five of its twenty cells individually differing from their published value by more than $10^{-4}$, so its re-run inference was not bit-identical." Drop the 0.0016 figure (the true worst case is smaller, i.e. the paper understates rather than overstates its own reproducibility) and drop any reference to a 'self-validation gate' -- the script that defines SANITY_TOL (analysis/experiments/rescore_nested_cv_corrected.py) was never run; the actual rescoring script (analysis/metrics/src/corrected_metrics_cv.py, committed) has no such gate.
- **Evidence:** Read main.tex:419: 'The two agree to $0.0016$ F1 at worst and to $3\times10^{-7}$ at the median, and the aggregate reproduces the published summary exactly.' Wrote a script diffing corrected_metrics.json['orig_all_f1'] (the re-run, buggy-matcher F1 from the actually-committed rescoring script, analysis/metrics/src/corrected_metrics_cv.py, commit 72f4d33) against cell_result.json['test_f1_all'] (the originally published per-cell F1) over all 100 nested-CV cells across the five run directories: max=0.000522 (5cv_esm2_adapter_only/outer2_inner3), median=3.148e-7 -- median matches the paper exactly, but the worst case (5.2e-4) is ~3x smaller than the paper's claimed 0.0016, in every reading I tried (all-segment F1 alone: max 5.22e-4; including peptide/propeptide sub-F1 and precision/recall: max 1.25e-3, still short of 0.0016). At the aggregate level, diffed nested_cv_summary.json['cv_f1_all_mean'] against nested_cv_summary_corrected.json['cv_orig_all_f1_mean'] for all five configs: four agree to ~1e-8-5e-8 (floating-point-level, i.e. genuinely 'exact'), but 5cv_esm2_adapter_only differs by 4.09e-5 (0.581628 vs 0.581669) -- so 'reproduces exactly' is false for one of five configs, though the discrepancy never surfaces as a printed number anywhere else in the paper. I also checked the finding's own claim about a 'self-validation gate': grepped and read analysis/experiments/rescore_nested_cv_corrected.py (SANITY_TOL=1e-4, used at line 199) -- this script exists and says what the finding quotes, BUT `git status` shows it is untracked/never-committed, and `find runs -iname cell_result_corrected.json` returns zero files anywhere in the repo, meaning this particular script was never actually run to produce the data in corrected_metrics.json. The data behind my diff instead comes from the committed, gate-less analysis/metrics/src/corrected_metrics_cv.py (commit 72f4d33, whose message claims 'recomputed orig f1_all reproduces every cell's published test_f1_all exactly' -- itself slightly wrong for the same 5 adapter cells). So the finding's numeric claims (worst 5.2e-4, median 3.15e-7, adapter aggregate off by 4.1e-5, same 5 specific cells >1e-4: outer2/inner3, outer4/inner1, outer0/inner2, outer3/inner0, outer1/inner2) all reproduce exactly, but its framing that these 5 cells 'exceed the rescore script's own self-validation gate' misattributes the 1e-4 threshold to a script that never touched this data (it's a threshold value that happens to exist in an unused sibling script, not a gate that was actually applied here).

### [minor] Table 4's 3Di row reports the peptide side of the trade-off as -0.04; the artifact says -0.033

- **Where:** 454 (the "Structural channel (3Di)" row of \Cref{tab:verdict})
- **As written:** Structural channel (3Di) & extra input & propep.\ $+0.02$ / pep.\ $-0.04$, net trade-off & no effect \\
- **Problem:** Recomputing the net-3Di contrast under the corrected matcher (esmc6b_3di_gated_boundary against its 3Di-zeroed control esmc6b_3di_zeroctrl, pooled over the two held-out folds, common protein set n = 2,322) gives peptide dF1 = -0.0335, which rounds to -0.03, not -0.04. The propeptide half is right (+0.0208) and so is the "no effect" verdict (net -0.0022, CI [-0.012, +0.008]). Only the peptide magnitude is inflated by one unit in the second decimal, and it is the number that makes the trade-off look asymmetric.
- **Correction:** Structural channel (3Di) & extra input & propep.\ $+0.02$ / pep.\ $-0.03$, net trade-off & no effect \\
- **Evidence:** Read main.tex:454 (table row: 'propep. +0.02 / pep. -0.04, net trade-off'). Loaded analysis/metrics/clean_regime_protfp.csv (per-protein tp/fn/fp, 7 models incl. esmc6b_3di_gated_boundary and esmc6b_3di_zeroctrl, folds {2,5}). Computed pooled micro-F1 on the common 2322-protein set: pep F1 gated_boundary=0.65502 vs zeroctrl=0.68851, delta=-0.03348 (rounds to -0.03, not -0.04); propep delta=+0.02082 (matches +0.02); all-tasks delta=-0.00220 (matches the 'net' claim). Cross-checked provenance: clean_regime_protfp.csv's fold-2 subset reproduces clean_split_modelselect.csv's corr_pep column exactly (0.69335 vs listed 0.6934 for zeroctrl; 0.69404 vs 0.694 for gated_boundary), confirming this is corrected-matcher data as the section's caption claims. Also reproduced the finding's cross-check: clean_split_modelselect.csv pep 0.694 vs 0.6934 (+0.0006) and clean_split_sealed_test.csv pep 0.6214 vs 0.6849 (-0.0635), averaging to -0.031. All three independent computations land at -0.03, none at -0.04.

### [minor] "on peptides alone it does not hold at all" is true of the mean but not of every fold

- **Where:** 421 (appendix C, last sentence)
- **As written:** per outer fold the full ordering holds in four folds of five after correction and three of five before it, and on peptides alone it does not hold at all.
- **Problem:** The two fold counts are exactly right. The peptide clause overstates by one fold: the mean ordering on peptides does fail (base 0.5632 < head 0.5891 but adapter 0.5804 < head, so base < head < adapter < both breaks), and it fails on four of the five outer folds — but it does hold on outer fold 1, where the corrected peptide F1s are 0.6312 < 0.6496 < 0.6594 < 0.6859. "Not at all" invites a reader to check and find a counterexample in a paragraph whose whole point is careful accounting of how often the ordering survives.
- **Correction:** ...and on peptides alone the mean ordering does not hold, surviving in only one fold of five (outer fold 1: $0.6312 < 0.6496 < 0.6594 < 0.6859$).
- **Evidence:** Read main.tex (appendix C, 'per outer fold the full ordering holds in four folds of five after correction and three of five before it, and on peptides alone it does not hold at all'). Loaded corrected_metrics.json from all outer{0..4}_inner{0..3} cells of runs/5cv_baseline_esm2, 5cv_esm2_boundary, 5cv_esm2_adapter_only, 5cv_esm2_full. First validated the paper's own method by reproducing its two other counts with corr_all_f1 and orig_all_f1 (mean over the 4 inner cells per outer fold, then check base<head<adapter<both): got exactly 4/5 (fails only outer4) for corrected and exactly 3/5 (fails outer2, outer4) for original -- both match the paper's stated counts precisely, confirming the aggregation method. Applying the identical method to corr_peptides_f1: outer0 0.6351/0.6795/0.6707/0.6853 (fails, adapter<head), outer1 0.6312/0.6496/0.6594/0.6859 (HOLDS), outer2 0.5574/0.5472/0.5568/0.5846 (fails), outer3 0.4713/0.5059/0.4928/0.5232 (fails), outer4 0.5209/0.5634/0.5223/0.5900 (fails) -- 1 of 5 folds holds, not 0. Overall means 0.5632/0.5891/0.5804/0.6138 match the run-level nested_cv_summary_corrected.json cv_corr_peptides_f1_mean values (mean-level failure is real, e.g. adapter<head), but the per-fold claim 'does not hold at all' is contradicted by outer fold 1, a clean counterexample.

### [nit] Three numbers are off by 0.001 from the artifact, all consistent with rounding twice through four decimals

- **Where:** 297 (recall delta), 286 and 292 (Table 1)
- **As written:** The adapter is mostly a recall effect: recall rises by $0.042$ against $0.013$ of precision.  ... $0.616 \pm 0.020$ ... $0.561 \pm 0.027$
- **Problem:** Adapter recall delta: 0.579163 - 0.537700 = 0.041463, which rounds to 0.041, not 0.042. Table 1 base precision: 0.615488 rounds to 0.615, not 0.616. Table 1 ESM-C recall: 0.560485 rounds to 0.560, not 0.561. All three become the printed value only if the artifact value is first rounded to four decimals (0.0415, 0.6155, 0.5605) and then to three, so this looks like one transcription habit rather than three independent slips. Every other cell of Table 1 (37 of 40 numbers) rounds correctly in one step.
- **Correction:** 0.041 (adapter recall gain, line 297); 0.615 (Table 1 base precision, line 286); 0.560 (Table 1 ESM-C recall, line 292)
- **Evidence:** Recomputed independently from runs/*/nested_cv_tolerance.json: adapter recall 0.579163, base recall 0.537700, delta=0.041463 -> rounds to 0.041 directly, but main.tex:297 prints 0.042. base precision 0.615488 -> rounds to 0.615 directly, but Table 1 (line 286) prints 0.616. ESM-C recall 0.560485 -> rounds to 0.560 directly, but Table 1 (line 292) prints 0.561. Tested the double-rounding hypothesis with Decimal/ROUND_HALF_UP: rounding each value to 4 decimals first (0.5792/0.5377 -> delta 0.0415; 0.6155; 0.5605) and then to 3 decimals reproduces the paper's printed 0.042/0.616/0.561 exactly. This is a reproducible two-step-rounding artifact, confirmed by direct computation.

### [nit] "Every gap widens as the tolerance tightens" is not monotone for two of the three variants

- **Where:** 301
- **As written:** Every gap widens as the tolerance tightens, from $+0.024$ to $+0.039$ for the boundary head, $+0.029$ to $+0.051$ for the adapter and $+0.058$ to $+0.086$ for the two together
- **Problem:** The endpoints are right, but the gap does not widen monotonically. The adapter's gap goes 0.0293 -> 0.0323 -> 0.0304 -> 0.0509 and the combined model's 0.0577 -> 0.0591 -> 0.0571 -> 0.0862, both dipping between +-2 and +-1. Only the boundary head is monotone (0.0241 -> 0.0263 -> 0.0269 -> 0.0391). The following clause ("each is nearly flat from $\pm3$ to $\pm1$") describes the real shape, so the opening generalisation overstates it.
- **Correction:** "Every gap is wider at an exact match than at ±3, from $+0.024$ to $+0.039$ for the boundary head, $+0.029$ to $+0.051$ for the adapter and $+0.058$ to $+0.086$ for the two together" -- replacing 'widens as the tolerance tightens' (which implies step-wise monotonicity that doesn't hold at ±2→±1 for two of the three variants) with a plain endpoint comparison, consistent with the sentence's own following clause.
- **Evidence:** Read main.tex:301 ('Every gap widens as the tolerance tightens, from +0.024 to +0.039 for the boundary head, +0.029 to +0.051 for the adapter and +0.058 to +0.086 for the two together, and each is nearly flat from ±3 to ±1 before opening up at the last step'). Read analysis/metrics/boundary_error_cv.json directly: abs_gap boundary {3:0.0241,2:0.0263,1:0.0269,0:0.0391} is monotone increasing; abs_gap adapter_only {3:0.0293,2:0.0323,1:0.0304,0:0.0509} dips at tol=1 (0.0323->0.0304); abs_gap full {3:0.0577,2:0.0591,1:0.0571,0:0.0862} also dips at tol=1 (0.0591->0.0571). So the literal step-by-step claim 'every gap widens' is false for 2 of 3 variants between ±2 and ±1. However the very next clause in the same sentence ('each is nearly flat from ±3 to ±1 before opening up at the last step') already describes this exact shape (flat/noisy until the last step, real widening only at the end) -- a ~0.002-0.002 dip is well inside 'nearly flat'. So a reader is not actually misled by the paragraph as a whole; only the opening four words, read in isolation, overstate monotonicity.

### [nit] Table 3's median-peptide-length row gives the 2022 column a value the 2022 data does not have

- **Where:** 363 (the last data row of \Cref{tab:dataset})
- **As written:** Median peptide length (residues) & 20--21 & 20--21 \\
- **Problem:** Computed from the two labeled_sequences.csv files: 2022 median peptide length is 21 and median propeptide length is 21; 2026 median peptide is 20 and median propeptide is 21. So "20--21" is a fair summary of the 2026 column (peptides 20, propeptides 21) but not of the 2022 column, where nothing is 20. The row label also says "peptide", so the range is presumably meant to cover both segment types — which is worth saying, since as printed the row's only job is to assert the two releases are identical on this axis, and they are not quite.
- **Correction:** Median peptide length (residues) & 21 & 20 \\
- **Evidence:** Read main.tex:363, tab:dataset row 'Median peptide length (residues) & 20--21 & 20--21'. Parsed data/uniprot_2022/labeled_sequences.csv and data/uniprot_2026/labeled_sequences.csv coordinate columns directly (regex over the '(start-end)' segment lists). 2022: 6372 peptide segments (matches table's Peptide-segments row exactly), median length 21.0; 8211 propeptide segments (matches table), median 21.0 -- both exactly 21, no 20 anywhere. 2026: 7431 peptide segments (matches table), median 20.0; 9140 propeptide segments (matches table), median 21.0. Verified all lengths already lie in [5,50] (min=5, max=50 in both files, 0 outliers), so no extra filtering was needed to match the table's segment counts, confirming these are the exact populations the table is built from. So '20--21' correctly describes the 2026 column (peptide=20, propeptide=21) but the 2022 column has no 20 in it at all (both peptide and propeptide medians are 21) -- the row's own label says 'peptide', and even under the generous 'peptide+propeptide' reading the 2022 cell should read 21, not a range including 20.

### [nit] The motif-clustering description says four residues per boundary; the code uses four residues per segment

- **Where:** 371-372 (appendix B, second paragraph)
- **As written:** obtained by $k$-means ($k=50$) on ESM-2 embeddings of the four residues
flanking each annotated boundary
- **Problem:** k=50 is right and the four-residue count is right, but they are not four residues per boundary. The clustering script takes, for a segment [s, e], the positions {s-2, s-1, e+1, e+2} — two residues outside the N-terminal boundary and two outside the C-terminal boundary — and concatenates their four embeddings into one 5120-d vector per *segment*, then clusters those. As written a reader reconstructs a two-residues-each-side window around a single cleavage site, which is a different feature and a different number of clustered items.
- **Correction:** ...on ESM-2 embeddings of the two residues flanking each end of an annotated segment, concatenated into one vector per segment
- **Evidence:** Read main.tex:371-372: 'k-means (k=50) on ESM-2 embeddings of the four residues flanking each annotated boundary.' Read analysis/dataset/src/flanking_motif_clusters.py in full: K=50 (line 41) and KMeans(n_clusters=args.k, ...) confirm k=50 is right. But flank_vector(emb, s, e) builds pos = [s-2, s-1, e+1, e+2] -- two residues outside the segment's start and two outside its end -- and np.concatenate(vecs) produces ONE 5120-d vector per SEGMENT (docstring: 'Concatenation order: [N-2,N-1,C+1,C+2] -> 4*1280=5120-d'), which the k-means then clusters (X = np.vstack(seg_vecs), one row per segment). A segment has two boundaries (start and end); the code takes 2 residues at each of those two boundaries, not 4 residues at a single boundary, and clusters one vector per segment, not per boundary. 'Four residues flanking each annotated boundary' reads as a 2-residue-each-side window around ONE cleavage site -- a different, half-sized feature clustering a different (larger, per-boundary) set of items than what the code does.

### [nit] Table 4's gated-adapter row is an ESM-C 6B measurement but, unlike its neighbours, does not say so

- **Where:** 456 (the "Gated adapter" row of \Cref{tab:verdict}); read against line 440
- **As written:** Gated adapter (pLM re-projection) & input adapter & $+0.022$, CI $[+0.008, +0.038]$ & helps \\
- **Problem:** The number is exactly reproducible, but it comes from an ESM-C 6B pair: runs/2026_esmc6b_adapter256_seqonly (embedding_dim 2580, 2560->256 re-projection, 3Di zeroed, seq-only) against runs/2026_esmc6b_boundary. The two rows directly above it are labelled "Boundary head on ESM-C 6B" and "Boundary head on ESM-2" precisely because the embedding matters here, and line 440 then counts this row among the three "later re-tested under nested cross-validation" — where the adapter was tested on ESM-2 only (1280->256). A reader comparing +0.022 to the nested-CV +0.029 will not know the two are on different embeddings.
- **Correction:** Gated adapter on ESM-C 6B (pLM re-projection) & input adapter & $+0.022$, CI $[+0.008, +0.038]$ & helps \\
- **Evidence:** Read main.tex:456 ('Gated adapter (pLM re-projection) & input adapter & $+0.022$, CI $[+0.008, +0.038]$ & helps') against the two rows above it (line 449-450) which are explicitly labelled 'Boundary head on ESM-C 6B' / 'Boundary head on ESM-2', and against line 212 ('1280 to 256 for ESM-2') and line 440 ('the adapter (Cref{sec:method})... later re-tested under nested cross-validation'). Read runs/2026_esmc6b_adapter256_seqonly/config.json: embedding_dim=2580, model='lstmcnncrf_gated3di_boundary', embeddings_dir='.../embeddings_esmc6b_3dizero' -- this is an ESM-C 6B run, not ESM-2. Paired it against runs/2026_esmc6b_boundary (embedding_dim=2560) using analysis/metrics/adapter256_perprotein_2026.csv and clean_regime_protfp.csv (model esmc6b_boundary), pooled tp/fn/fp over both tasks on the common 2361 proteins: F1 0.68744 vs 0.66520, delta=+0.02224, 5000-resample bootstrap CI [+0.0076, +0.0380] -- reproduces the table's '+0.022, CI [+0.008,+0.038]' to 3 decimals. Per-fold breakdown: fold2=-0.00098, fold5=+0.04482, matching main.tex:429's single-split '-0.001 on one fold and +0.045 on the other' for 'the isolated gated adapter'. So the row is confirmed to be an ESM-C 6B (2560/2580-dim) measurement, unlike its two neighbouring 'Boundary head' rows which disclose their embedding in the label, and unlike the method section's adapter (1280->256 on ESM-2, the one actually re-tested at +0.029 under nested CV per line 186).


---

## 2. Numbers — dismissed on review

- **"a confident $+0.03$" contradicts the same table's own row text and the fold-level numbers three paragraphs earlier** (440 (appendix D.2, first paragraph), against line 451 (the table row) and line 429) — Read main.tex:440 ('...overlapping intervals, where this table records a confident +0.03') against line 451 ('$+0.03$, unstable across folds') and line 429 ('+0.074 on one and -0.011 on the other'). Reproduced clean_split_paired_triage.csv rows 'plain pLM: 6B vs ESM2': fold2=0.0742, fold5=-0.0111 -- these match line 429 exactly, so the paper already states the sign-flip elsewhere; nothing is hidde


---

## 3.consistency — 19 candidates, NOT adjudicated

Lens: consistency. Confirmations reported by the lens: Verified against the artifacts and found correct (no action needed):

TABLE 2, every cell. All five rows reproduce runs/*/nested_cv_tolerance.json to the digit shown, except the one ESM-C recall noted above: base P 0.615488/R 0.537700/F1 0.572523+-0.025254, 0.539872, 0.460652, 0.345350; boundary 0.672512/0.537855/0.596663+-0.026084, 0.566220, 0.487568, 0.384462; adapter 0.628111/0.579163/0.601807+-0.026185, 0.572154, 0.491059, 0.396202; full 0.667344/0.598328/0.630193+-0.020985, 0.598926, 0.517797, 0.431585; ESM-C 0.619596/…/0.587692+-0.015705, 0.558306, 0.483478, 0.370541. The whole Growth column is right (+0.014972, +0.021568, +0.028565, +0.010022).

HEADLINE GAINS. +0.024 / +0.029 / +0.05

### [major] Appendix Table 5 promises per-cell standard deviations that it does not contain

- **Where:** main.tex:482-484 (text); main.tex:495 \input{figures/tolerance_table.tex}; figures/tolerance_table.tex
- **As written:** \Cref{tab:tolerance} repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level
standard deviation on every cell.
- **Problem:** The included table has no standard-deviation columns at all. Its header is `\textbf{Configuration} & $\pm3$ & $\pm2$ & $\pm1$ & exact & \textbf{gap growth}` and every cell is a bare mean (e.g. `ESM-2, base & $0.573$ & $0.540$ & $0.461$ & $0.345$ & $+0.000$`). Table 5 is therefore an exact duplicate of the four F1 columns plus the growth column of Table 2 and adds nothing, while the sentence that introduces it states its only reason to exist. The stds do exist in the sibling artifact `figures/tolerance_table.csv` (tol3_std 0.0253, tol2_std 0.0255, tol1_std 0.0165, tol0_std 0.0079 for the base row), so the .tex generator dropped them.
- **Correction:** Either regenerate `figures/tolerance_table.tex` with the `*_std` columns from `tolerance_table.csv` (which already holds them for all five configurations at all four tolerances), or delete Table 5 and the sentence, since without the stds it repeats Table 2 verbatim.
- **Evidence:** figures/tolerance_table.tex contains 5 data rows x 5 numeric columns, no `\pm`; figures/tolerance_table.csv has columns tol3_std, tol2_std, tol1_std, tol0_std populated for every row.

### [major] The stated criterion for "no confirmed improvement" would also reject both additions

- **Where:** main.tex:305
- **As written:** The intervals overlap, so an embedding five times wider buys no confirmed improvement on its own, while either architectural addition on the narrower one does.
- **Problem:** Interval overlap does not separate the cases the sentence contrasts. Using Table 2's own numbers: ESM-C 6B $0.588\pm0.016$ overlaps base $0.573\pm0.025$; but the boundary head $0.597\pm0.026$ and the adapter $0.602\pm0.026$ overlap the base by the same test, and by wider margins. Only the combined model ($0.630\pm0.021$, lower edge 0.609 above the base's upper edge 0.598) is separated by non-overlap. The clause "while either architectural addition on the narrower one does" is therefore false under the criterion the same sentence invokes.
- **Correction:** State the criterion that actually separates them, which the artifacts support: per outer fold, the boundary head and the adapter beat the base on 5/5 folds, whereas ESM-C 6B beats it on only 4/5 (fold 4: 0.5919 vs 0.5985). E.g. "the intervals overlap and ESM-C 6B loses to ESM-2 on one of the five outer folds, whereas each architectural addition wins on all five".
- **Evidence:** runs/*/nested_cv_summary_corrected.json, per_outer_corr_all_f1: base {0.5585, 0.5943, 0.5378, 0.5735, 0.5985}; boundary {0.5928, 0.6130, 0.5597, 0.5892, 0.6286}; adapter {0.6222, 0.6256, 0.5607, 0.5947, 0.6059}; esmc6b {0.5749, 0.6105, 0.5710, 0.5902, 0.5919}.

### [major] The adapter's parameter arithmetic does not reconcile with its own components, and "trainable" is the wrong word for 235,113

- **Where:** main.tex:212 (and 210, 340)
- **As written:** the projection itself holds 331{,}008 parameters, partly offset because the convolutions downstream now read 256 channels instead of 1280, for a net $+287{,}558$.
- **Problem:** 331,008 is correct, but the stated net implies an offset of only 43,450. The single downstream width change is the first convolution, which goes from Conv1d(1280, 32, k=3) = 122,912 params to Conv1d(256, 32, k=3) = 24,608, an offset of 98,304; the second convolution is unchanged (it reads the 128-wide BiLSTM output either way). The true net is 331,008 - 98,304 = +232,704. The published 287,558 is exactly the difference of the two saved state_dicts (522,671 - 235,113), which double-counts an aliased BiLSTM (`self.biLSTM = self.backbone.biLSTM` makes 50,176 elements appear twice in the adapter checkpoint) and includes the 4,678-parameter boundary head that the adapter-only run carries at `boundary_state_scale: 0.0`. The same conflation makes 235,113 a state_dict element count, not a trainable-parameter count: 10,403 of those are the CRF's non-trainable `_constraint_mask`/`_constraint_start_mask`/`_constraint_end_mask` buffers. The project's own artifact records the trainable figure as 224,710.
- **Correction:** Use +232,704 for the adapter's net cost (or +237,382 if the dormant boundary head is deliberately included), and 224,710 for the base model's trainable parameters at both L210 and L340. If the state_dict figures are kept, say "235,113 stored tensor elements (224,710 trainable)".
- **Evidence:** Instantiated from `src/models/crf_models.py` with the run configs: LSTMCNN(1280)=214,112 vs LSTMCNN(256)=115,808 (delta 98,304); GatedResidualConvSplitProjector(seq_only=True)=331,008; LSTMCNNCRF base trainable=224,710; adapter model trainable=462,092 (delta 237,382, of which 4,678 is the head). torch.load of runs/5cv_baseline_esm2/outer0_inner1/model.pt = 235,113 elements over 20 tensors including 10,403 of CRF constraint buffers; runs/5cv_esm2_adapter_only/outer0_inner1/model.pt = 522,671 with `feature_extractor.biLSTM.*` duplicating `feature_extractor.backbone.biLSTM.*` (50,176). analysis/metrics/clean_model_params.csv: baseline_esm2,224710.

### [major] The ESM-C 6B row was trained and tested on a different protein set from the ESM-2 rows, and the paper does not say so

- **Where:** main.tex:224 and Table 2 (main.tex:292)
- **As written:** 8{,}897 of them carry ESM-2 embeddings and enter the five folds used here.
- **Problem:** Table 1 presents 8,897 (1558/2572/1263/2025/1479) as the fold composition of the study, and Table 2's ESM-C 6B row is compared head-to-head with the ESM-2 rows, with the abstract calling it "statistically indistinguishable from its ESM-2 counterpart under the same protocol". The ESM-C 6B cells were in fact run on `graphpart_assignments_5motif.esmc6bcovered.csv`, which holds 8,999 proteins with fold sizes 1580/2600/1273/2054/1492 - 123 proteins the ESM-2 runs never see, and 21 ESM-2 proteins the ESM-C runs never see. The test partitions therefore differ by ~1.1%, so the paper's only cross-embedding comparison is not on identical data. (The folds themselves are consistent: all 8,876 shared proteins carry the same fold id in both files, which is what makes this a disclosure gap rather than a broken comparison.)
- **Correction:** Add one clause where the ESM-C row is introduced, e.g. "the ESM-C 6B cells run on the 8,999 proteins with ESM-C embeddings, which share fold assignments with the 8,897 ESM-2 proteins on the 8,876 in both", and say which count Table 1 describes.
- **Evidence:** runs/5cv_esmc6b_plain/*/config.json: `partitioning_file: data/uniprot_2026/graphpart_assignments_5motif.esmc6bcovered.csv`, `embeddings_dir: data/uniprot_2022/embeddings/embeddings_esmc6b`, `embedding_dim: 2560`; the ESM-2 runs use `..._5motif.esm2covered.csv`. Row counts: esm2covered 8,897; esmc6bcovered 8,999; intersection 8,876 with 0 fold-id disagreements.

### [major] Appendix D quotes two different values for the same ESM-2 single-split baseline (0.621 and 0.572)

- **Where:** main.tex:472
- **As written:** scored $0.558$ and $0.567$ against $0.621$ for the frozen baseline under the same protocol.
- **Problem:** Everything else in Appendix D uses an ESM-2 single-split baseline of 0.572: Figure 5's bottom row is "ESM-2 (baseline) 0.572", Figure 6's left point is 0.572, Figure 10's ESM-2 curve ends at 0.572 at 100% of the training folds, and Table 4's "+0.03" for the ESM-C 6B swap is 0.604 - 0.572. The 0.621 comes from a different dataset release: the LoRA runs and their frozen comparator are on uniprot_2022, while the rest of the appendix is on the uniprot_2026 rebuild. The phrase "under the same protocol" tells the reader the numbers are comparable to the rest of the appendix, and they are not - a reader subtracting gets "LoRA costs 0.06" against a baseline that appears nowhere else in the paper.
- **Correction:** Say which release these runs are on, e.g. "...against $0.621$ for the frozen baseline in the same comparison (these three runs are on the 2022 release, so the numbers are not on the scale of Fig. 5)". Alternatively quote the LoRA deficits as deltas (-0.063 and -0.054) rather than absolutes.
- **Evidence:** runs/esm2_lora_lstmcnncrf/config.json and runs/train_run_esm2/config.json both point at `data/uniprot_2022/labeled_sequences.csv`; analysis/metrics/corrected_metrics.csv: esm2_lora_lstmcnncrf corr_all_f1 0.558080, esm2_lora_lstmcnncrf_r4_last2_qv 0.567131, train_run_esm2 0.620803. runs/2026_baseline_esm2/config.json points at `data/uniprot_2026/...`; analysis/metrics/clean_2026_table.csv: baseline_esm2 f1 0.5711 (0.572 on the figure's common subset).

### [minor] The abstract says combining the additions costs something; the data say it gains

- **Where:** main.tex:148
- **As written:** combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base, with little loss from applying them together
- **Problem:** "little loss from applying them together" asserts sub-additivity. The two effects are +0.02414 and +0.02928, summing to 0.05342, and the combined effect is +0.05767 - the joint model beats the sum of the parts by 0.004. The body states this correctly ("together they give $+0.058$, close to the sum of the parts", L297), so the abstract contradicts Section 5 on the one mechanistic claim it makes about combining them.
- **Correction:** "...combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base, slightly more than the sum of their separate effects."
- **Evidence:** runs/*/nested_cv_summary_corrected.json cv_corr_all_f1_mean: base 0.5725232, boundary 0.5966630 (+0.0241398), adapter 0.6018073 (+0.0292840), full 0.6301931 (+0.0576698); 0.0241398 + 0.0292840 = 0.0534238 < 0.0576698.

### [minor] "five times wider" contradicts the widths the paper states elsewhere

- **Where:** main.tex:305
- **As written:** so an embedding five times wider buys no confirmed improvement on its own
- **Problem:** ESM-C 6B embeddings are 2560-dimensional and ESM-2's are 1280, so the embedding is two times wider, not five. The paper's own numbers say so: Section 3 describes the adapter as reducing "1280 to 256 for ESM-2" and Table 4's compression row is "ESM-C 6B compression 2560$\to$256" with "$10\times$ narrower input". Five is neither the width ratio (2x) nor the parameter ratio (6B vs 650M, about 9x).
- **Correction:** "an embedding twice as wide" (or "a pLM ten times larger", if the intent was model size rather than embedding width).
- **Evidence:** runs/5cv_esmc6b_plain/*/config.json `embedding_dim: 2560`; runs/5cv_baseline_esm2/*/config.json `embedding_dim: 1280`; main.tex:212 and main.tex:455.

### [minor] Contributions and Conclusion state a localization result for the boundary head that Section 5 explicitly declines to state

- **Where:** main.tex:186 and main.tex:310
- **As written:** the head buys precision, the adapter buys recall, and both place segment ends more precisely, measured on the segments a variant and the base model both find.
- **Problem:** On the paired, detection-controlled set the Results paragraph reports the head's effect and then withholds it: "The boundary head moves it from $0.622$ to $0.653$, but only four folds agree and the spread ($\pm0.040$) exceeds the effect, so we report the direction and not the magnitude" (L303). The contributions bullet and the Conclusion's "both place segment ends more precisely" carry no such qualification, so the two summaries claim more than the section they summarize. The same paragraph also draws the sharper contrast itself: "the adapter is the better sharpener".
- **Correction:** Qualify both: "the adapter also places segment ends more precisely, and the head does so in the same direction on four of five folds" - or drop "both" and attribute the confirmed sharpening to the adapter and the combined model, which are positive on 5/5 folds.
- **Evidence:** Recomputed with `env/bin/python analysis/metrics/src/boundary_error_cv.py`: boundary head per-outer delta(exact share) +0.018 +0.008 -0.001 +0.098 +0.032 (mean +0.031 +- 0.040, 4/5 positive); adapter +0.078 +0.048 +0.017 +0.084 +0.060 (mean +0.057 +- 0.027, 5/5); both +0.055 +0.092 +0.040 +0.139 +0.078 (mean +0.081 +- 0.038, 5/5).

### [minor] Table 2 and Figure 2 are cited before Table 1 and Figure 1

- **Where:** main.tex:214 (first \Cref{tab:main-results}), main.tex:201 (first \Cref{fig:arch}), main.tex:240 (first \Cref{tab:genus}), main.tex:299 (first \Cref{fig:hook})
- **As written:** The two additions intervene at different points, so we test them separately and together, giving the $2\times2$ factorial in \Cref{tab:main-results}.
- **Problem:** Float numbers follow placement, but the text's citation order is inverted for both float classes. Figure 2 (the architecture scheme) is first cited at L201, Figure 1 (the hook) not until L299 in Results. Table 2 (main results) is first cited at L214 in Method and again at L233, while Table 1 (the genus table) is first cited at L240. A reader following the text meets "Table 2" and "Figure 2" before either "Table 1" or "Figure 1" exists in the narrative.
- **Correction:** Add an early pointer to Figure 1 (the hook figure sits on page 2 with no sentence referring to it until page 5 - one clause in the Introduction, e.g. after "The head is zero-initialized...", would fix it), and either move the genus table after the results table or cite it earlier, in the paragraph that introduces the folds.
- **Evidence:** main.aux: `\newlabel{fig:hook}{{1}{2}...}`, `\newlabel{fig:arch}{{2}{3}...}`, `\newlabel{tab:genus}{{1}{5}...}`, `\newlabel{tab:main-results}{{2}{5}...}`.

### [minor] Table 3 and Figure 3 are never referenced from any text

- **Where:** main.tex:353 (\label{tab:dataset}) and main.tex:396 (\label{fig:datadist})
- **As written:** \label{tab:dataset}
- **Problem:** Neither label appears in any \Cref/\ref anywhere in the file. The dataset-composition table (Table 3) and the dataset-distribution figure (Figure 3) sit in Appendix B with no sentence pointing at them, even though Section 4.1 states exactly the counts Table 3 tabulates (8,449 / 9,619 / 1,178) and would naturally cite it. The supplementary figures 5-12 are at least introduced collectively at L500; Table 3 and Figure 3 have no such introduction.
- **Correction:** Cite Table 3 from L224 ("...of which 1,178 were absent in 2022 (\Cref{tab:dataset})") and Figure 3 from the segment-length/motif paragraph at L368-373.
- **Evidence:** Full inventory of cross-references in main.tex: tab:main-results x10, sec:app-old-protocol x6, sec:method x5, sec:results x4, sec:eval-protocol x4, sec:dataset x4, tab:verdict x3, tab:genus x2, sec:related x2, sec:matching x2, sec:eval-metric x2, tab:tolerance x1, sec:intro, sec:eval, sec:app-tolerance, sec:app-related, sec:app-problem, fig:hook, fig:folds, fig:arch. No tab:dataset, no fig:datadist.

### [minor] Figure 11's caption describes only panel (a); panel (b) shows a different quantity and goes unmentioned

- **Where:** main.tex:547
- **As written:** F1 against training-data volume at two tolerances, single-split protocol. The $\pm3$ curve is flat for the strong model ($0.676$ to $0.694$, overlapping intervals) while the exact-match curve rises monotonically ($0.455$ to $0.505$)
- **Problem:** The PNG has two labelled panels. Panel (a) is titled "ESM-C 6B + boundary + gated(256), 3Di zeroed: F1 at ±3 and ±0" and matches the caption's numbers (0.675 to 0.694 at ±3, 0.455 to 0.505 at ±0). Panel (b) is titled "Retained F1 share at an exact match (F1@±0 / F1@±3)" and plots a ratio for three models, including the ESM-2 baseline, which the caption never mentions - the reader is given no key to the second half of the figure, nor to why the ESM-2 curve is non-monotonic there (0.435, 0.554, 0.599, 0.543, 0.552).
- **Correction:** Extend the caption to name panel (b), e.g. "(a) F1 at two tolerances for the strong model; (b) the retained F1 share at an exact match for all three, where the ESM-2 baseline stays well below the ESM-C models at every data size."
- **Evidence:** figures/datascale_tolerance.png rendered: panel (a) axis labels "F1" vs "number of proteins in the training set"; panel (b) y-axis "retained F1 share (F1@±0 / F1@±3)" with three legend entries and in-plot note "bands: 95 % bootstrap CI (both panels); test set: pooled folds {2} and {5}".

### [minor] Table 4's "Gated adapter" row is not gated, is a different factorial cell, and does not name its embedding

- **Where:** main.tex:456 (row) and main.tex:440 (the sentence that ties it to Section 3)
- **As written:** Gated adapter (pLM re-projection) & input adapter & $+0.022$, CI $[+0.008, +0.038]$ & helps \\
- **Problem:** Three problems in one row. (i) Naming: Section 3 describes the adapter as "LayerNorm, Linear, GELU" with no gate, and the code agrees - in `GatedResidualConvSplitProjector` the entire gate (`self.gate`, `self.gate_ln`, `struct_to_seq`) is inside `if not seq_only:` and is never built for these runs, so "gated" survives only from the model-class name. Figure 5's caption then uses "gated" for something else again ("The top row is a gated-projector control at full width, not the configuration carried into the main text"), and Figures 5, 7, 8, 10, 11, 12 label rows "gated(256)", "gated(2560)" and "3Di zeroed", terms defined nowhere in the text. (ii) L440 says this row is "the adapter (\Cref{sec:method})" re-tested under nested CV, but the run behind it has `boundary_state_scale: 1.0`, i.e. adapter *on top of* the boundary head, whereas the nested-CV adapter row is the adapter alone (`boundary_state_scale: 0.0`). The comparable nested-CV cell is full minus boundary = 0.630 - 0.597 = +0.033. (iii) The row does not name its embedding, while the two rows above it do ("on ESM-C 6B", "on ESM-2"); it is ESM-C 6B, which a reader will not guess and which is what makes the abstract's "the additions on that embedding... are reported in the appendix" and the Limitations' "on ESM-C 6B they have single-split evidence" true.
- **Correction:** Rename the row "Input adapter on ESM-C 6B (on top of the boundary head)" and drop "gated" from the table; either define gated(256)/gated(2560)/3Di zeroed once in the appendix or relabel the supplementary figures in the paper's vocabulary.
- **Evidence:** src/models/crf_models.py:1207 `if not seq_only:` guards the gate construction, and the seq_only forward path is `seq_projector -> out_ln` only. runs/2026_esmc6b_adapter256_seqonly/config.json: `gated_seq_only: true`, `boundary_state_scale: 1.0`, `embeddings_dir: data/uniprot_2022/embeddings/embeddings_esmc6b_3dizero`. runs/5cv_esm2_adapter_only/*/config.json: `gated_seq_only: true`, `boundary_state_scale: 0.0`. analysis/metrics/adapter256_perprotein_2026.csv model column is `adapter256_seqonly`. Figure 5's in-plot note: "dashed rule - gated adapter: +0.022 (isolated by ablation)".

### [minor] Confirmed nested-CV material is filed inside the appendix that disclaims everything in it

- **Where:** main.tex:479-486 (subsection sec:app-tolerance, numbered D.4)
- **As written:** None of the numbers below should be read with the same confidence as \Cref{tab:main-results}.
- **Problem:** That sentence is Appendix D's preamble (L425), and D is titled "Earlier single-split protocol: unconfirmed observations". But D.4 "Complete tolerance sweep" is nested-CV material: it repeats Table 2's sweep and supplies the 26,730 / 29,051 / 29,132 paired segment counts that Section 5 (L303) cites as the evidence for its localization claim. Its own caption says "$5\times4$ nested cross-validation, corrected matcher". D.5 then resumes with "The figures below are from the same earlier development stage as \Cref{tab:verdict}", so D.4 is bracketed on both sides by single-split framing. A reader who honours the preamble discounts the confirmed evidence that Results depends on.
- **Correction:** Move D.4 out of Appendix D into its own appendix section (or into Appendix C next to the matcher discussion), or add a one-line note at the head of D.4 stating that this subsection alone reports nested-CV results.
- **Evidence:** main.aux: `\newlabel{sec:app-tolerance}{{D.4}{14}{Complete tolerance sweep}...}` inside `\newlabel{sec:app-old-protocol}{{D}{12}{Earlier single-split protocol: unconfirmed observations}...}`; main.tex:490-491 caption "Segment F1 by boundary-match tolerance, $5\times4$ nested cross-validation".

### [nit] Adapter recall gain stated as +0.042; the artifact and the paper's own table both give 0.041

- **Where:** main.tex:297
- **As written:** The adapter is mostly a recall effect: recall rises by $0.042$ against $0.013$ of precision.
- **Problem:** The adapter's recall at ±3 is 0.579163 against the base's 0.537700, a rise of 0.041463, which rounds to 0.041. Table 2's own rounded entries give the same: 0.579 - 0.538 = 0.041. The precision figure in the same sentence (0.628111 - 0.615488 = 0.012623 -> 0.013) is right, as are the head's +0.057/+0.0002 and the combined +0.052/+0.061.
- **Correction:** "recall rises by $0.041$ against $0.013$ of precision."
- **Evidence:** runs/5cv_esm2_adapter_only/nested_cv_tolerance.json cv_tol3_all_recall_mean 0.579163; runs/5cv_baseline_esm2/nested_cv_tolerance.json cv_tol3_all_recall_mean 0.537700.

### [nit] ESM-C 6B recall in Table 2 rounds the wrong way

- **Where:** main.tex:292
- **As written:** --- & --- & $0.620 \pm 0.017$ & $0.561 \pm 0.027$ & $0.588 \pm 0.016$ & $0.558$ & $0.483$ & $0.371$ & $+0.010$ \\
- **Problem:** The recall mean is 0.560485, which rounds to 0.560, not 0.561. Every other cell in that row is correct (precision 0.619596->0.620, std 0.017079->0.017, F1 0.587692->0.588, std 0.015705->0.016, 0.558306->0.558, 0.483478->0.483, 0.370541->0.371, growth 0.010022->+0.010), as is every cell of the four ESM-2 rows.
- **Correction:** $0.560 \pm 0.027$
- **Evidence:** runs/5cv_esmc6b_plain/nested_cv_tolerance.json: cv_tol3_all_recall_mean 0.560485, cv_tol3_all_recall_std 0.027435.

### [nit] Cross-reference for the 50-residue cap points at a section that does not state it

- **Where:** main.tex:340
- **As written:** (101 states in total for the 50-residue cap used here, \Cref{sec:dataset})
- **Problem:** Section 4.1 (sec:dataset) never mentions the cap; it defers with "Segment-length filtering, motif balancing and the full composition of the rebuild are given in \Cref{sec:app-problem}". The 5/50 bounds are stated in Appendix B (L368-369): "Segments shorter than 5 or longer than 50 residues are dropped: the CRF's state count fixes the upper bound". A reader chasing the reference lands on a section that forwards them somewhere else.
- **Correction:** \Cref{sec:app-problem}
- **Evidence:** main.tex:224 (sec:dataset body) vs main.tex:368-373 (sec:app-problem body).

### [nit] "reproduces the published summary exactly" is not exact for one of the five configurations

- **Where:** main.tex:419
- **As written:** The two agree to $0.0016$ F1 at worst and to $3\times10^{-7}$ at the median, and the aggregate reproduces the published summary exactly.
- **Problem:** Four of five configurations reproduce to <=1e-7, but 5cv_esm2_adapter_only does not: the re-scored aggregate is 0.581669 against the published 0.581628 (4.1e-5), and every one of its five per-outer values differs in the fourth decimal (0.6006/0.6069/0.5415/0.5700/0.5893 re-scored vs 0.6005/0.6070/0.5414/0.5701/0.5892 published). The median claim is exact (2.9e-7 over 300 per-cell comparisons); the worst-case claim is conservative (the true maximum is 0.00125, better than the stated 0.0016).
- **Correction:** "...and the aggregate reproduces the published summary to $4\times10^{-5}$ or better."
- **Evidence:** runs/5cv_esm2_adapter_only/nested_cv_summary.json cv_f1_all_mean 0.5816281 vs nested_cv_summary_corrected.json cv_orig_all_f1_mean 0.5816691; per-cell max |orig - published| across all 100 cells x 3 tasks = 0.001252, median 2.93e-7.

### [nit] Figure 2 labels the encoder output [L x 128]; the configuration in the text makes it 64

- **Where:** main.tex:206 (caption for figures/architecture_scheme.jpg)
- **As written:** The boundary head reads the encoder output and adds a position-specific correction to the per-state CRF emissions before decoding.
- **Problem:** The figure's zoom panel labels that encoder output "Features [L x 128]". For the configuration the paper reports - "32 convolutional filters" (L340) - the second convolution emits 32*2 = 64 channels, so the features are [L x 64]. This is not cosmetic: the boundary head's 4,678 parameters (L184, L210) only come out right at 64 (LayerNorm 128 + Linear 64->64 4,160 + Linear 64->6 390), and at 128 the head would cost 8,966. The 128 in the figure is the class default (n_filters=64), not the trained model.
- **Correction:** Relabel to "Features [L x 64]" (and "Emissions [L x 3]" stays correct).
- **Evidence:** runs/5cv_baseline_esm2/outer0_inner1/model.pt: `feature_extractor.conv2.weight (64, 128, 5)` and `features_to_emissions.weight (3, 64)`; runs/5cv_esm2_boundary/.../model.pt: `boundary_to_state.net.0.weight (64,)`, `net.2.weight (64,64)`, `net.5.weight (6,64)`, summing to 4,678.

### [nit] Figure 3's genus counts do not match Table 1's, and neither caption says which population it counts

- **Where:** main.tex:395 (caption) vs main.tex:244 (Table 1 caption)
- **As written:** Composition of the rebuilt 2026 dataset: segment length, type, and genus distributions.
- **Problem:** Figure 3's genus panel gives Conus 921, Homo 457, Cyriopagopus 305, Lycosa 175; Table 1 gives Conus 714, Cyriopagopus 293, Lycosa 163 and a Homo total of 432 (71+81+66+130+84). Both are internally correct - Figure 3 counts the 9,619 proteins before homology partitioning (its peptide/propeptide legend, n=7431 and n=9140, matches Table 3 exactly), Table 1 counts the 8,897 that enter the folds - but neither caption says so, and Table 3 is the only place "Counts are before homology partitioning" appears. A reader comparing the two sees a 30% discrepancy on Conus with nothing to explain it.
- **Correction:** Add "before homology partitioning" to Figure 3's caption, mirroring Table 3's wording.
- **Evidence:** figures/data_distributions.png panel (b) bar labels 921/461/457/305/300/284/217/175/166/132 and panel (a) legend "peptides (n=7431)" / "propeptides (n=9140)"; main.tex:361-362 Table 3 rows 7431 and 9140; main.tex:251-255 Table 1.


---

## 3.claims — 16 candidates, NOT adjudicated

Lens: claims. Confirmations reported by the lens: Verified against the artifacts and correct:

TABLE 1 (line 269-295), every cell. All five rows of precision, recall and the four F1 columns reproduce runs/*/nested_cv_tolerance.json to the digit shown (base 0.6155/0.5377/0.5725/0.5399/0.4607/0.3453; boundary 0.6725/0.5379/0.5967/0.5662/0.4876/0.3845; adapter 0.6281/0.5792/0.6018/0.5722/0.4911/0.3962; full 0.6673/0.5983/0.6302/0.5989/0.5178/0.4316; esmc6b 0.6196/0.5605/0.5877/0.5583/0.4835/0.3705). The growth column arithmetic is right too: +0.015, +0.022, +0.029, +0.010 are exactly gap(exact) - gap(±3).

Section 5 precision/recall decomposition (line 297) is exact: head +0.0570 P and +0.0002 R; adapter +0.0415 R against +0.0126 P; combined +

### [blocker] The ESM-2 base row was trained under bf16 autocast; every other row was trained in fp32, so every effect in Table 1 is architecture + numerics

- **Where:** 269-295 (Table 1); asserted as isolated effects at 148, 186, 297
- **As written:** both additions outperform the base architecture on ESM-2 embeddings individually ($+0.024$ and $+0.029$ F1) and combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base
- **Problem:** The 20 cells of runs/5cv_baseline_esm2 all carry "amp": true with "amp_dtype": "bf16"; the 20 cells of 5cv_esm2_boundary, 5cv_esm2_adapter_only, 5cv_esm2_full and 5cv_esmc6b_plain all carry "amp": false. src/train_loop_crf.py:504-520 and 566-635 feed that flag straight into torch.amp autocast for the training and validation passes, so the baseline's weights were produced under mixed precision and the four comparison rows' were not. The baseline config is also missing the driver keys "K" and "resume" that all four other configs carry, i.e. it came off an older version of the training script. Nothing in the paper mentions a precision or pipeline difference (grep for bf16/amp/fp32/mixed in main.tex returns nothing). Every headline number -- +0.024, +0.029, +0.058, and the ESM-2 vs ESM-C 6B contrast -- is therefore the architectural change confounded with a numerics-and-pipeline change, and the paper's causal reading ("the head buys precision", "they act on different error types") does not survive that.
- **Correction:** Either retrain the ESM-2 base with amp:false to match the variants, or state the difference explicitly in section 4.3 and demote the causal language in section 5 to "the base configuration as trained differs from the variants in numerical precision as well as architecture; we cannot separate the two here." The cheap partial check is to retrain the base for one or two outer folds with amp:false and report the delta.
- **Evidence:** for f in runs/5cv_*/outer*_inner*/config.json -> amp True x20 for 5cv_baseline_esm2, False x20 for the other four runs. src/train_loop_crf.py:504 `use_amp = getattr(args, "amp", False)`, :510 `scaler = GradScaler(...)`, :620-635 pass use_amp/amp_dtype into the train and valid passes. Base config keys lack "K" and "resume".

### [major] "The intervals overlap, so no confirmed improvement" is applied to ESM-C 6B but not to the additions, which fail the same test; the paired design that would settle it is never used

- **Where:** 305
- **As written:** The intervals overlap, so an embedding five times wider buys no confirmed improvement on its own, while either architectural addition on the narrower one does.
- **Problem:** Applied consistently, the paper's own criterion rules out both additions. Table 1 gives base $0.573\pm0.025$, boundary head $0.597\pm0.026$, adapter $0.602\pm0.026$: 0.573+0.025 = 0.598 sits inside both variants' lower bounds (0.571 and 0.576). Only the combined row (0.630±0.021) is separated by the overlap test. The comparison is also unnecessarily weak: all five configurations were run on the same five outer folds, so the design is paired and the marginal mean±std throws that away. Paired over the shared folds the picture is defensible but different from the one stated -- head +0.024±0.008 (5/5 folds, t-test p=0.0023), adapter +0.029±0.021 (5/5, p=0.036), combined +0.058±0.016 (5/5, p=0.0013), ESM-C 6B +0.015±0.014 (4/5, p=0.075). ESM-C is weaker, not categorically different.
- **Correction:** Replace the interval-overlap reasoning with the paired per-fold difference the design already supports: report mean±std of the per-outer-fold delta, the sign count, and a paired test. Then the sentence becomes "the head and the adapter are positive on all five outer folds (p=0.002 and p=0.036 paired); ESM-C 6B is positive on four of five (p=0.075)", which is honest and stronger for the additions.
- **Evidence:** Per-outer ±3 F1 recomputed from runs/*/outer*_inner*/tolerance_metrics.json (mean over the 4 inner cells of each outer fold, matching nested_cv_tolerance.json exactly). base [0.5585, 0.5943, 0.5378, 0.5735, 0.5985]; boundary [0.5928, 0.6130, 0.5597, 0.5892, 0.6286]; adapter [0.6222, 0.6256, 0.5607, 0.5947, 0.6059]; full [0.6394, 0.6496, 0.6026, 0.6132, 0.6461]; esmc6b [0.5749, 0.6105, 0.5710, 0.5902, 0.5919]. scipy.stats.ttest_rel against base gives p = 0.0023 / 0.0359 / 0.0013 / 0.0750.

### [major] The fold-non-exchangeability argument measures the wrong variance component; the paper's own per-fold vectors show fold composition contributes nothing to the effects

- **Where:** 187, 260, 312
- **As written:** This puts a floor under how small an effect a single split can resolve. The base architecture scores $0.538$, $0.559$, $0.574$, $0.594$ and $0.599$ on the five outer folds: a range of $0.061$, which is two and a half times the $+0.024$ effect of the boundary head
- **Problem:** A single split evaluates both models on the same held-out fold, so the fold's absolute difficulty is common to both and cancels in the difference. The across-fold range of absolute scores therefore puts no floor on a paired comparison, and the paper's own data proves it: the boundary head's per-outer-fold delta is +0.034, +0.019, +0.022, +0.016, +0.030 -- positive on every fold, sd 0.008, so any one outer fold taken alone would have resolved it at roughly the right size. A one-way variance decomposition over the 20 cells makes it sharper: for the head's paired delta the between-outer-fold sd is 0.0000 while the within-outer (inner split / training) sd is 0.0267; for the adapter 0.019 vs 0.019, for the combined 0.013 vs 0.019, for ESM-C 0.010 vs 0.021. Fold composition dominates the absolute scores (base: 0.024 between vs 0.015 within) and contributes little or nothing to the differences. What nested CV averages down here is inner-split and training noise, not taxonomic fold imbalance. The same mis-attribution carries the conclusion (line 312) and contribution (iv) (line 187).
- **Correction:** Rewrite the argument around the quantity that actually varies: the per-cell spread of the paired difference (sd ≈ 0.02-0.027 across the 20 cells, collapsing to 0.008-0.021 across the 5 outer-fold means). Say that a single training run of each configuration is the unreliable unit, and that averaging 20 cells is what buys the resolution. Keep Table 2 as evidence that folds differ, but stop using the 0.061 absolute range as the bound on a paired effect.
- **Evidence:** Per-outer deltas above, from runs/*/outer*_inner*/tolerance_metrics.json. ANOVA over the 20 cells (4 inner per outer): var_between = max((MSB-MSW)/4, 0). Paired delta vs base: boundary sd_between=0.0000 sd_within=0.0267 (per-cell sd 0.0248, 18/20 cells positive); adapter 0.0189/0.0185; full 0.0128/0.0189; esmc6b 0.0097/0.0206. Absolute ±3 F1: base sd_between=0.0241 sd_within=0.0151.

### [major] "What capacity buys here is boundary precision rather than more segments found" is contradicted by the precision/recall columns of the table it sits under

- **Where:** 305
- **As written:** It does show the same tolerance signature as the additions, $+0.015$ at $\pm3$ against $+0.025$ at an exact match, so what capacity buys here is boundary precision rather than more segments found.
- **Problem:** Table 1's own columns say the opposite. ESM-C 6B vs ESM-2 base at ±3: precision 0.620 vs 0.616 (+0.0041, positive on 3 of 5 folds, paired p=0.80) and recall 0.561 vs 0.538 (+0.0228, 4 of 5 folds, p=0.10). At an exact match: precision +0.0190 (3/5, p=0.29), recall +0.0293 (5/5, p=0.0065). At both tolerances the ESM-C advantage is carried by recall -- "more segments found" -- which is exactly what the sentence denies. The supporting "tolerance signature" is also the weakest of the four rows: its growth is +0.010 with a fold-level sd of 0.023, negative on two of five outer folds (per fold: +0.014, -0.005, -0.021, +0.038, +0.023; one-sample p=0.39). This row is also the only one with no paired-localization check behind it, so it rests on the growth number alone.
- **Correction:** Drop the mechanism sentence, or invert it to match the table: ESM-C 6B's ±3 advantage over ESM-2 comes from recall (+0.023) with essentially no precision gain (+0.004), and its exact-match advantage likewise (recall +0.029, precision +0.019, the latter not fold-consistent). If the tolerance-signature claim is kept, it needs the fold spread reported alongside it.
- **Evidence:** Per-outer precision/recall from runs/*/outer*_inner*/tolerance_metrics.json: tol3 precision base [0.6058,0.6100,0.6285,0.5917,0.6414] vs esmc [0.6104,0.6321,0.6206,0.6388,0.5960]; tol3 recall base [0.5190,0.5797,0.4710,0.5568,0.5619] vs esmc [0.5442,0.5908,0.5297,0.5496,0.5881]. tol0 recall delta [+0.035,+0.008,+0.030,+0.039,+0.036]. Growth per fold from analysis/metrics/src/boundary_error_cv.py logic recomputed per outer fold.

### [major] The contributions list and the conclusion state flatly what section 5 explicitly hedges about the boundary head

- **Where:** 186 and 310 vs 303
- **As written:** the head buys precision, the adapter buys recall, and both place segment ends more precisely, measured on the segments a variant and the base model both find
- **Problem:** Section 5 (line 303) says of the head on the same paired set: "The boundary head moves it from $0.622$ to $0.653$, but only four folds agree and the spread ($\pm0.040$) exceeds the effect, so we report the direction and not the magnitude." That is a correctly hedged null-adjacent result. The intro contribution (line 186) and the conclusion (line 310, "and both place segment ends more precisely, so their advantage widens as the match tolerance tightens") drop the hedge and assert it for both blocks. A reader who reads only the abstract, contributions and conclusion -- which is most of them -- gets a claim the evidence section refuses to make.
- **Correction:** Match the summary layer to the evidence layer: "the adapter also places segment ends more precisely (+0.057±0.027, five folds of five); the head points the same way but not consistently enough to quantify." Same edit in the conclusion.
- **Evidence:** analysis/metrics/boundary_error_cv.json, paired block: boundary head exact_base 0.6219 -> exact_variant 0.6530, per-outer exact-share delta +0.018 +0.008 -0.001 +0.098 +0.032 (mean +0.031 ± 0.040, 4/5 folds positive); adapter +0.078 +0.048 +0.017 +0.084 +0.060 (+0.057 ± 0.027, 5/5); full +0.055 +0.092 +0.040 +0.139 +0.078 (+0.081 ± 0.038, 5/5). Reproduced by running analysis/metrics/src/boundary_error_cv.py.

### [major] The growth column carries no uncertainty, the promised appendix standard deviations do not exist in the table file, and "none of them is flat" holds for one row of four

- **Where:** 271-295 (growth column), 167 (Fig 1 caption), 482-496 (Table 4)
- **As written:** (c) The absolute gap to the ESM-2 base at each tolerance; a flat line would mean the curve is a parallel translation of the baseline, and none of them is flat.
- **Problem:** Growth is a difference of differences and its fold-level spread is comparable to or larger than the values reported: boundary +0.015 ± 0.023 (4/5 folds positive, p=0.21), adapter +0.022 ± 0.017 (5/5, p=0.047), combined +0.029 ± 0.030 (4/5, p=0.10), ESM-C +0.010 ± 0.023 (3/5, p=0.39). Only the adapter row is fold-consistent, so "none of them is flat" is unsupported for three of the four. Separately, section F.4 promises the missing uncertainty -- "\Cref{tab:tolerance} repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level standard deviation on every cell" -- but texs/ai4dd/figures/tolerance_table.tex, which is \input into that table, has five columns of bare means and no standard deviations at all. The one table a reader would consult to check the growth claim does not contain what the text says it contains.
- **Correction:** Regenerate tolerance_table.tex with the per-fold standard deviations (they are already in runs/*/nested_cv_tolerance.json as cv_tol{t}_all_f1_std), add a spread and a fold sign-count to the growth column, and soften the Figure 1(c) caption to the one row it holds for. The paired paragraph at line 303 partially rescues the additions; it does not cover the ESM-C row, which rests on growth alone.
- **Evidence:** cat texs/ai4dd/figures/tolerance_table.tex -> columns are Configuration, ±3, ±2, ±1, exact, gap growth; every cell a single number. Growth per outer fold, computed from tolerance_metrics.json: boundary [0.0057,-0.0030,0.0057,0.0540,0.0126]; adapter [0.0114,0.0152,0.0026,0.0425,0.0362]; full [0.0022,0.0348,-0.0033,0.0702,0.0389]; esmc [0.0143,-0.0053,-0.0205,0.0382,0.0234].

### [minor] Checkpoint and epoch selection ran on the buggy validation matcher, a caveat the rescoring script says lives in the appendix but which the appendix does not contain

- **Where:** 231 and 417
- **As written:** everything reported here is re-scored with a corrected matcher (\Cref{sec:eval-metric})
- **Problem:** analysis/experiments/rescore_nested_cv_corrected.py states, under "CAVEAT THIS SCRIPT DOES NOT FIX": "This only re-scores the FINAL test-set evaluation. It cannot retroactively change which epoch was selected as 'best' during training -- that early-stopping selection was itself based on buggy validation F1 (compute_all_metrics on the validation split, same bug). A run whose true best epoch differs under the corrected metric will not be captured by re-scoring alone. See the paper appendix for how we describe this." The paper appendix does not describe it: the words epoch, early stopping and selection appear in section C nowhere (grep -n "epoch" main.tex hits only lines 429 and 472, both in appendix D). So "everything reported here is re-scored" is true of the scoring and false of the model selection that produced the checkpoints being scored.
- **Correction:** Add one sentence to section C: the correction was applied at scoring time only; the checkpoint chosen for each cell was selected on validation F1 computed with the uncorrected matcher, so a cell whose optimal epoch differs under the corrected metric is not recovered by rescoring. This also belongs in Limitations.
- **Evidence:** analysis/experiments/rescore_nested_cv_corrected.py, docstring section "CAVEAT THIS SCRIPT DOES NOT FIX". grep -ni "epoch\|early" texs/ai4dd/main.tex returns lines 301 (unrelated), 429, 472 only.

### [minor] "Nudges boundaries inward" is unsupported and the nearest evidence points the other way

- **Where:** 303
- **As written:** on ESM-2 the head filters spurious segments and nudges boundaries inward as a side effect
- **Problem:** Nothing in the artifacts supports a direction for the head's boundary movement, and the detection-controlled measurement points against "inward". Under the strict paired ±3 gate in analysis/metrics/segment_quality_cv.json the head's mean absolute start displacement gets *worse* by +0.011 residues with 0 of 5 outer folds improved, while the end displacement improves by -0.044 (4/5). The only signed numbers available (peptide dstart -0.161 -> -0.431, dend +0.074 -> +0.431; propeptide -0.101 -> -0.387, +0.417 -> +0.525, with dstart = pred_start - true_start) say the head pushes both boundaries *outward*, lengthening segments -- but those are per-model means over each model's own matched segments, the survivorship-biased quantity, so they cannot carry the claim either. Neither reading supports "inward".
- **Correction:** Delete the clause, or replace it with what the paired data does show: the head's gain is concentrated in the end boundary (paired |Δend| -0.044 residues, 4/5 folds) with no improvement at the start (+0.011, 0/5), and its precision gain comes from predicting 15% fewer propeptide segments (1531.8 -> 1297.4 per cell) at an unchanged peptide count (1033.6 -> 1029.5).
- **Evidence:** analysis/metrics/segment_quality_cv.json, models.5cv_esm2_boundary.paired_vs_baseline.tol3: d_abs_dstart_mean mean +0.0110 per_outer [0.0151,0.0059,0.0026,0.0073,0.0242] n_outer_improved 0; d_abs_dend_mean mean -0.0444, improved 4. metrics.peptides_dstart_signed_mean / dend_signed_mean and peptides_n_pred / propeptides_n_pred for base vs boundary. Sign convention at segment_quality_cv.py:223-224 `dstart[task].append(ps - ts)`.

### [minor] Figure 1(a) labels the protein's C-terminus an "annotated cleavage site", contradicting the generator's own selection criterion

- **Where:** 159-167
- **As written:** with the $\pm3$ acceptance window shaded around each annotated cleavage site and the baseline's displacement in residues marked above it
- **Problem:** The worked example is G3ETQ3 in cell outer1_inner0. Its sequence is 71 residues and its annotated peptide is [43, 71], so the second of the two boundaries the panel marks is the end of the chain, not a cleavage site. The generator's own comment claims the opposite: "A frog antimicrobial-peptide precursor whose propeptide and mature peptide are adjacent, so both of the baseline's errors fall on cleavage sites rather than on the ends of the chain." One of the two does not. The ±3 window drawn there is one-sided in practice (residues 72-74 do not exist), so half of the figure's headline illustration is a case where the error can only run one way. This is not a niche case: across the grid 28.6% of true peptide ends and 27.8% of true propeptide ends are the C-terminus.
- **Correction:** Pick an example whose two errors are both at interior boundaries, or keep this one and say in the caption that the second marked position is the chain terminus. The interior-only displacement statistics already exist (peptides_dend_interior_abs_mean etc. in segment_quality_cv.json) if a stratified version is wanted.
- **Evidence:** runs/5cv_baseline_esm2/outer1_inner0/segments.json.gz for G3ETQ3: peptides true [[43,71]] pred [[43,68]], propeptides true [[23,40]] pred [[23,42]]; len(labeled_sequences.loc['G3ETQ3','sequence']) = 71. Comment at analysis/metrics/src/generators/fig_hook_cv.py:46-48. segment_quality_cv.json peptides_frac_end_at_c_terminus 0.2856, propeptides 0.2782.

### [minor] Parameter counts: 235,113 is params plus buffers, not trainable parameters, and the adapter's net cost is +232,704, not +287,558

- **Where:** 210, 212, 340
- **As written:** It costs 4{,}678 parameters on top of a trainable stack of 235{,}113, next to a frozen pLM of 650 million.
- **Problem:** Instantiating the base model from runs/5cv_baseline_esm2/outer0_inner1/config.json gives 224,710 trainable parameters (features_to_emissions 195 + crf 10,403 + feature_extractor 214,112). The quoted 235,113 is 224,710 plus the CRF's 10,403 non-trainable buffers, so it is the total including buffers mislabelled as trainable; the same number is repeated at line 340 ("giving 235{,}113 trainable parameters for the base model"). The adapter figure at line 212, "the projection itself holds 331{,}008 parameters ... for a net $+287{,}558$", gets the projector right (331,008 measured) but not the net: the backbone saving from reading 256 channels instead of 1280 is 214,112 - 115,808 = 98,304, so the net is 331,008 - 98,304 = +232,704 (or +237,382 if the disabled boundary head is counted). +287,558 does not correspond to any counting I could reproduce. The boundary head's 4,678 is exact.
- **Correction:** Use 224,710 trainable (or say "235,113 parameters including the CRF's transition buffers") and +232,704 for the adapter's net cost.
- **Evidence:** get_model on each cell's config: base trainable=224710 total=224710 buffers=10403; adapter-only trainable=462092 with feature_extractor.projector=331008, feature_extractor.backbone=115808, boundary_to_state=4678; base feature_extractor=214112.

### [minor] The ESM-C 6B row is scored on a different protein set from the ESM-2 rows, contradicting "the same protocol" and the 8,897 count

- **Where:** 148, 224, 267
- **As written:** $8{,}897$ of them carry ESM-2 embeddings and enter the five folds used here
- **Problem:** The ESM-2 rows use data/uniprot_2026/graphpart_assignments_5motif.esm2covered.csv (8,897 proteins, folds 1558/2572/1263/2025/1479). runs/5cv_esmc6b_plain uses ...esmc6bcovered.csv (8,999 proteins, folds 1580/2600/1273/2054/1492); 123 proteins are present only for ESM-C and 21 only for ESM-2. Table 1 therefore compares 0.588 and 0.573 on non-identical test sets while the abstract says "under the same protocol" and section 4.1 states a single 8,897-protein set. I re-scored both runs at ±3 with the corrected matcher restricted to the 8,876 shared proteins per cell: ESM-C 0.5881±0.0164 against base 0.5722±0.0250, a paired delta of +0.0159±0.0128 versus the +0.0152 reported. So the conclusion does not move -- this is a reporting defect, not a wrong number.
- **Correction:** One sentence in section 4.1: the ESM-C 6B run covers 8,999 proteins because 123 sequences have ESM-C embeddings but no ESM-2 ones (and 21 the reverse); restricted to the 8,876 shared proteins the ESM-C figure is 0.588 and the gap is +0.016, unchanged.
- **Evidence:** graphpart_assignments_5motif.esm2covered.csv 8897 {0:1558,1:2572,2:1263,3:2025,4:1479}; ...esmc6bcovered.csv 8999 {0:1580,1:2600,2:1273,3:2054,4:1492}; set difference 123 / 21, shared 8876. Shared-set rescore over the 20 paired cells (35,504 protein-cells) with analysis.errors.src.error_analysis.match_protein at tol 3: base per-outer [0.5587,0.5928,0.5376,0.5734,0.5986], esmc [0.5742,0.6109,0.5709,0.5885,0.5959].

### [minor] "No effect at all" overstates the single-split result: the same head measured +0.013 there, and the appendix figures show the bond-loss variant instead

- **Where:** 260, 453, 463
- **As written:** Under an earlier single-split protocol we ran during development, that same modification measured as no effect at all, with a confidence interval covering zero (\Cref{sec:app-old-protocol}). The modification did not change; the resolution did.
- **Problem:** analysis/metrics/clean_2026_table.csv, the single-split pooled table, gives baseline_esm2 F1 0.5711 and esm2_boundary F1 0.5845 -- a point estimate of +0.013 for the identical modification (runs/2026_esm2_boundary has bond_loss_lambda 0.0 and boundary_state_scale 1.0, the same as the nested-CV cells). That is about half the nested-CV +0.024, same sign, with a bootstrap CI covering zero. "No effect at all" describes the CI, not the estimate, and it makes the rhetorical contrast look larger than it is. Separately, the ESM-2 boundary variant carried in the appendix figures is the wrong one: clean_tol_true.csv / clean_tol_pred.csv, which drive Figures 6 and 8, contain esm2_boundary_bond (runs/2026_esm2_boundary_bond, bond_loss_lambda 0.02 -- the auxiliary loss Table 5 itself calls "harmful") and not esm2_boundary. A reader checking the "≈0" row against the figures is looking at a different model.
- **Correction:** State the single-split point estimate: "+0.013 with a bootstrap interval covering zero", and rewrite the punchline as "the single split saw the effect at half its size and could not separate it from zero". Regenerate the appendix scoreboard and tolerance figures from esm2_boundary rather than esm2_boundary_bond, or label the row as the bond-loss variant.
- **Evidence:** analysis/metrics/clean_2026_table.csv: baseline_esm2 f1 0.571105, esm2_boundary 0.584487, esm2_boundary_bond 0.582590 (n_test 2325). runs/2026_esm2_boundary/config.json bond_loss_lambda 0.0; runs/2026_esm2_boundary_bond/config.json bond_loss_lambda 0.02. clean_tol_true.csv model list: baseline_esm2, esm2_boundary_bond, esmc6b_*, esmc_600m, esmc_6b -- no esm2_boundary.

### [nit] "Little loss from applying them together" describes a sub-additivity the data does not show

- **Where:** 148
- **As written:** combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base, with little loss from applying them together
- **Problem:** The joint gain (+0.0577) exceeds the sum of the parts (+0.0241 + 0.0293 = +0.0534), so the interaction is +0.0042, if anything slightly super-additive and statistically indistinguishable from zero (per outer fold -0.017, +0.005, +0.020, +0.003, +0.010; mean +0.0042 ± 0.0136). "Little loss" implies a measured loss; there is none. The body's version at line 297, "close to the sum of the parts", is accurate.
- **Correction:** Match the body: "and combine additively to +0.058", or "with no measurable interaction (+0.004 ± 0.014)".
- **Evidence:** Per-outer ±3 F1 from tolerance_metrics.json; interaction = full - boundary - adapter + base computed fold by fold.

### [nit] The five per-fold baseline scores are listed sorted, not in fold order, next to a table that labels folds f0-f4

- **Where:** 260
- **As written:** The base architecture scores $0.538$, $0.559$, $0.574$, $0.594$ and $0.599$ on the five outer folds
- **Problem:** The actual per-fold sequence is 0.559, 0.594, 0.538, 0.574, 0.599 for f0 through f4. The quoted list is the same five numbers sorted ascending, but the phrasing "on the five outer folds" invites a reader to map 0.538 to f0 -- and Table 2 on the same page labels its columns f0-f4 and invites exactly that cross-reference (0.538 is in fact f2, the smallest fold; 0.559 is f0). The range of 0.061 is correct either way.
- **Correction:** Either state them in fold order or say "sorted": "scores between 0.538 and 0.599 across the five outer folds (f0-f4: 0.559, 0.594, 0.538, 0.574, 0.599)".
- **Evidence:** runs/5cv_baseline_esm2/nested_cv_tolerance.json, per_outer_tol3_all_f1 = {0: 0.5585, 1: 0.5943, 2: 0.5378, 3: 0.5735, 4: 0.5985}.

### [nit] "Their spread exceeds every effect measured on them" holds only under the range reading, by 0.003

- **Where:** 187, 312
- **As written:** homology-aware folds are not exchangeable, their spread exceeds every effect measured on them
- **Problem:** The only spread the paper reports is the standard deviation across outer folds, 0.025 for the base row, which is smaller than the +0.058 combined effect. The claim survives only if "spread" means the range (0.061 vs 0.058), a margin of 0.003 that the sentence's rhetorical weight cannot support -- and "every effect" then rests on which of two undefined spread measures the reader picks.
- **Correction:** Name the statistic and the comparison: "the range of base scores across folds (0.061) is larger than any single effect measured here" -- and, given the previous finding about the wrong variance component, this sentence is probably better cut than repaired.
- **Evidence:** Table 1 reports ±0.025 for the base row; range from per_outer_tol3_all_f1 is 0.5985 - 0.5378 = 0.0607; largest effect 0.6302 - 0.5725 = 0.0577.

### [nit] "On peptides alone it does not hold at all" is false for one of the five folds

- **Where:** 421
- **As written:** per outer fold the full ordering holds in four folds of five after correction and three of five before it, and on peptides alone it does not hold at all
- **Problem:** The preceding clause counts folds, so "does not hold at all" reads as "in zero folds". On peptides the ordering base < boundary head < adapter < combined does hold on outer fold 1 (0.6312 < 0.6496 < 0.6594 < 0.6859). It fails on the other four and on the mean (0.5632, 0.5891, 0.5804, 0.6138, where the head is ahead of the adapter), which is the point being made -- the count is just wrong.
- **Correction:** "and on peptides alone it holds in one fold of five, and not on the mean, where the head is ahead of the adapter (0.589 against 0.580)".
- **Evidence:** Per-outer tol3_peptides_f1 from runs/*/outer*_inner*/tolerance_metrics.json: base [0.6351,0.6312,0.5574,0.4713,0.5209]; boundary [0.6795,0.6496,0.5472,0.5059,0.5634]; adapter [0.6707,0.6594,0.5568,0.4928,0.5223]; full [0.6853,0.6859,0.5846,0.5232,0.5900].


---

## 3.bibliography — 7 candidates, NOT adjudicated

Lens: bibliography. Confirmations reported by the lens: Every \cite key in main.tex resolves to a refs.bib entry (25 distinct keys used, all defined; the apparent gap for 'ESM-family' was a regex artifact from the entry's '@article {ESM-family,' spacing, not a missing key). All citations in main.tex use \citep exclusively, in the 'Name~\citep{key}' pattern (e.g. 'DeepPeptide~\citep{deeppeptide2023}', 'ProP~\citep{duckert2004prop}'), which reads correctly as a parenthetical citation after a named subject -- no instance calls for \citet instead. ankh2023 is the only entry formally typed as arXiv/@misc besides the ESM Cambrian blog and Hayes-preprint issue found above; checked and it remains an unpublished preprint (arXiv 2301.06568) as of this revi

### [blocker] bpfun2025 has a placeholder journal name and fabricated author surnames

- **Where:** refs.bib:283-288 (rendered at main.bbl:234-239, from main.tex:196,338 \citep{...,bpfun2025})
- **As written:** author  = {Yang, S. and Shen, H. and Zhang, L.},
  title   = {{BPFun}: a deep learning framework for bioactive peptide function prediction using multi-label strategy by transformer-driven and sequence rich intrinsic information},
  journal = {VERIFY-JOURNAL},
  year    = {2025}
- **Problem:** The journal field is a literal unfilled placeholder, and it renders that way in the compiled bibliography: main.bbl:239 reads '\newblock \emph{VERIFY-JOURNAL}, 2025.' -- so page 6+ of the built PDF's References section cites a journal called 'VERIFY-JOURNAL'. The author surnames are also wrong: none of Yang/Shen/Zhang match the real paper's authors.
- **Correction:** The real paper (same title, correctly matched) is: Lun Zhu, Hao Sun, Sen Yang, 'BPFun: a deep learning framework for bioactive peptide function prediction...', BMC Bioinformatics, vol. 26, article 187, 2025, DOI 10.1186/s12859-025-06190-5. Replace journal='BMC Bioinformatics', volume=26, pages/article=187, doi=10.1186/s12859-025-06190-5, and author={Zhu, Lun and Sun, Hao and Yang, Sen}.
- **Evidence:** Crossref lookup on DOI 10.1186/s12859-025-06190-5 (via WebFetch of https://api.crossref.org/works/10.1186/s12859-025-06190-5) returned: Full Author List: Lun Zhu, Hao Sun, Sen Yang; Journal: BMC Bioinformatics; Volume: 26; Article Number: 187; Published Year: 2025 -- same title as the bib entry. Confirmed the placeholder renders in the artifact via `grep -n -A6 bpfun main.bbl` in texs/ai4dd, output: '\newblock \emph{VERIFY-JOURNAL}, 2025.'

### [major] graphpart2023 author list contains two people who did not write the paper, misspells a real co-author's surname, drops a true co-author, and the title has a spurious hyphen

- **Where:** refs.bib:119-124 (rendered at main.bbl:192-198, cited from main.tex:196,224,342)
- **As written:** title={Graph-Part: homology partitioning for biological sequence analysis},
  author={Teufel, Felix and Refsgaard, Magn{\'u}s Halld{\'o}r and Madsen, Christian Garde and Marcatili, Paolo and Nielsen, Henrik and Winther, Ole and Armenteros, Jos{\'e} Juan Almagro},
- **Problem:** The real GraphPart paper (NAR Genomics and Bioinformatics 5(4):lqad088, 2023, DOI 10.1093/nargab/lqad088 -- the same DOI-matching venue/volume/pages as the bib entry) has six authors: Felix Teufel, Magnús Halldór Gíslason, José Juan Almagro Armenteros, Alexander Rosenberg Johansen, Ole Winther, Henrik Nielsen. The bib entry instead lists 'Refsgaard, Magnús Halldór' (wrong surname -- the real second author is Gíslason, not Refsgaard) and two names that are not GraphPart authors at all, 'Madsen, Christian Garde' and 'Marcatili, Paolo', while dropping the real co-author Alexander Rosenberg Johansen entirely. This is not a paraphrase issue: it renders as fabricated authorship in the built PDF's reference list (main.bbl:192-195: 'Felix Teufel, Magnús Halldór Refsgaard, Christian Garde Madsen, Paolo Marcatili, Henrik Nielsen, Ole Winther, and José Juan Almagro Armenteros.'). The title also has a spurious hyphen: every official source gives the title as 'GraphPart' (one word), not 'Graph-Part'; this too renders into the PDF (main.bbl:196: 'Graph-part: homology partitioning...').
- **Correction:** Replace author={Teufel, Felix and G{\'i}slason, Magn{\'u}s Halld{\'o}r and Armenteros, Jos{\'e} Juan Almagro and Johansen, Alexander Rosenberg and Winther, Ole and Nielsen, Henrik} and title={GraphPart: homology partitioning for biological sequence analysis}.
- **Evidence:** Cross-checked independently against the University of Copenhagen Research Portal, PMC (PMC10578201 via WebFetch), and Oxford Academic (academic.oup.com/nargab/article/5/4/lqad088/7318077 via WebFetch), all agreeing: authors Felix Teufel, Magnús Halldór Gíslason, José Juan Almagro Armenteros, Alexander Rosenberg Johansen, Ole Winther, Henrik Nielsen; title 'GraphPart: homology partitioning for biological sequence analysis'. Notably, the two wrong bib names -- 'Refsgaard' and 'Madsen, Christian [Toft]' -- are exactly the real co-authors of the adjacent deeppeptide2023 entry (refs.bib:2, 'Refsgaard, Jan Christian' and 'Madsen, Christian Toft'), which this session separately confirmed correct against fteufel.github.io/publication/deeppeptide -- suggesting the graphpart2023 author field was contaminated from that neighboring entry rather than looked up.

### [major] ESM-family (Hayes et al.) is cited as an unpublished 2024 bioRxiv preprint though it has since been published in Science

- **Where:** refs.bib:28-39 (cited from main.tex:338)
- **As written:** elocation-id = {2024.07.01.600583},
	year = {2024},
	doi = {10.1101/2024.07.01.600583},
	publisher = {Cold Spring Harbor Laboratory},
	...
	journal = {bioRxiv}
- **Problem:** This is exactly the case the task flags: a work cited as a preprint that has since been formally published. The bib entry still points to the bioRxiv doi/version even though the paper has been in a peer-reviewed journal since February 2025 -- more than half a year before this September 2026 submission.
- **Correction:** Update to the published version: Hayes et al., 'Simulating 500 million years of evolution with a language model', Science 387(6736):850-858, 21 Feb 2025, DOI 10.1126/science.ads0018 (journal=Science, volume=387, number=6736, pages=850--858, year=2025).
- **Evidence:** Crossref lookup on DOI 10.1126/science.ads0018 (via WebFetch) returned: Journal: Science; Volume: 387; Issue: 6736; Pages: 850-858; Published Date: February 21, 2025 -- same author list and title as the bioRxiv preprint currently cited.

### [major] ESM-family is cited alongside the ESM Cambrian blog post as if it supports a claim about ESM Cambrian, but the cited paper is about a different model (ESM3)

- **Where:** main.tex:338
- **As written:** ESM-2~\citep{lin2023esm2}, the earlier ESM line~\citep{esm2_2019}, ESM Cambrian~\citep{esm_cambrian_blog_2024,ESM-family}, and Ankh~\citep{ankh2023}.
- **Problem:** The sentence lists ESM Cambrian as one item in a survey of pLM families and cites two references for it. But the ESM-family bib entry's own abstract (refs.bib:35) says 'We present ESM3, a frontier multimodal generative language model that reasons over the sequence, structure, and function of proteins' -- this is the ESM3 paper, a distinct, generative multimodal model from EvolutionaryScale, not the ESM Cambrian/ESM-C encoder-representation line (300M/600M/6B) that esm_cambrian_blog_2024 actually describes and that this paper's own experiments use as 'ESM-C 6B'. Pairing the ESM3 paper with the ESM Cambrian blog implies it corroborates the ESM Cambrian claim; it does not describe that model at all.
- **Correction:** Either drop ESM-family from this citation (esm_cambrian_blog_2024 alone already correctly supports the ESM Cambrian claim, as used identically at main.tex:196), or, if the ESM family of models generally is meant to be surveyed, add a separate clause naming ESM3 explicitly rather than folding the Hayes et al. citation into the ESM Cambrian item.
- **Evidence:** refs.bib:29-38 (ESM-family entry) title/abstract describe ESM3 only ('We present ESM3...'); web search independently confirms Hayes et al. as the ESM3 paper (Science 387(6736):850-858), a separate release from ESM Cambrian/ESM-C (announced by a December 2024 EvolutionaryScale blog post, per esm_cambrian_blog_2024 and corroborating WebSearch results describing ESM-C as trained at 300M/600M/6B scale).

### [minor] Three bib entries truncate the author list with 'and others', one of them dropping the senior/corresponding author

- **Where:** refs.bib:161-170 (song2018prosperous), refs.bib:194-203 (li2020procleave), refs.bib:273-281 (zhang2024deepbp)
- **As written:** author  = {Song, Jiangning and others}  [song2018prosperous]
author  = {Li, Fuyi and Leier, Andr{\'e} and Liu, Quanzhong and Wang, Yanan and Xiang, Dongxu and Akutsu, Tatsuya and others}  [li2020procleave]
author  = {Zhang, M. and Zhou, J. and Wang, X. and others}  [zhang2024deepbp]
- **Problem:** Each of these drops real co-authors under an 'and others' stub instead of listing them. For li2020procleave the dropped names include Jiangning Song, who is the senior/last author of that paper (and the corresponding/senior author across this whole cluster of protease-cleavage papers by the same lab, cf. song2012prosper, li2020deepcleave, li2023prosperousplus, all correctly spelled out in full elsewhere in the same bib file) -- so the one paper in the group missing his name is exactly this one. For zhang2024deepbp the dropped authors are Xun Wang and Fang Ge.
- **Correction:** Fill in the full author lists: song2018prosperous -- Song J, Li F, Leier A, Marquez-Lago T, Akutsu T, Haffari G, et al. (per PROSPERous publication record); li2020procleave -- add Webb, Geoffrey I., Smith, A. Ian, Marquez-Lago, Tatiana, Li, Jian, and Song, Jiangning; zhang2024deepbp -- add Wang, Xun and Ge, Fang.
- **Evidence:** WebSearch results (Monash University publication record) give Procleave's full author list as 'Fuyi Li, Andre Leier, Quanzhong Liu, Yanan Wang, Dongxu Xiang, Tatsuya Akutsu, Geoffrey I. Webb, A. Ian Smith, Tatiana Marquez-Lago, Jian Li, and Jiangning Song.' Crossref (DOI 10.1186/s12859-024-05974-5) gives DeepBP's full author list as 'Ming Zhang, Jianren Zhou, Xiaohua Wang, Xun Wang, Fang Ge.'

### [nit] Two refs.bib entries are never cited anywhere in main.tex

- **Where:** refs.bib:49-54 (dima2024), refs.bib:56-66 (alphafold3_2024)
- **As written:** @article{dima2024, ... }
@article{alphafold3_2024, ... }
- **Problem:** Both entries (a protein-sequence-generation diffusion paper and the AlphaFold 3 paper) have no corresponding \cite/\citep/\citet anywhere in main.tex, and there is no \nocite either, so they will not appear in the compiled reference list at all -- dead weight in the .bib file.
- **Correction:** Either cite them somewhere relevant (e.g. AlphaFold 3 could support a passing mention of structure-based methods) or remove the unused entries from refs.bib.
- **Evidence:** `grep -oE '\\\\cite[tp]?\\{[^}]+\\}' main.tex | ...` extracted all 25 distinct cite keys used in main.tex; `comm -23` against the 28 keys defined in refs.bib leaves exactly dima2024 and alphafold3_2024 as bib-only entries. `grep -n nocite main.tex` returned no matches, so neither is pulled in that way either.

### [nit] GraphPart is credited with 'establishing the benchmark' when its own paper is a general partitioning method, not a benchmark/dataset paper

- **Where:** main.tex:196
- **As written:** DeepPeptide~\citep{deeppeptide2023} is, to our knowledge, the only model that segments a precursor into typed peptide and propeptide spans, and it also established the homology-partitioned benchmark used here~\citep{graphpart2023}, so we take both its architecture and its data pipeline as our starting point.
- **Problem:** The sentence's subject (the thing that 'established the ... benchmark') is DeepPeptide, and \citep{graphpart2023} is attached right after 'benchmark' as if GraphPart itself is a citation for that benchmark. GraphPart (Teufel et al., NAR Genomics and Bioinformatics 2023) is a general-purpose sequence-partitioning algorithm/tool paper, not a paper that describes or introduces this specific peptide/propeptide benchmark -- DeepPeptide is the one that applied GraphPart to build the benchmark, which the sentence itself says. This reading is defensible as a compressed 'via GraphPart' aside, but it is easy to misread as GraphPart being a benchmark paper.
- **Correction:** Rephrase for clarity, e.g. '...and it also established the benchmark used here, homology-partitioned with GraphPart~\citep{graphpart2023}, so we take...' to make clear GraphPart is the partitioning tool DeepPeptide used, not the benchmark's own source.
- **Evidence:** graphpart2023's real title/abstract (per academic.oup.com/nargab/article/5/4/lqad088 and refs.bib itself) describes a homology-partitioning algorithm for arbitrary datasets, with no peptide/propeptide-specific benchmark content; the peptide/propeptide benchmark itself is what deeppeptide2023 constructed (as main.tex:224 and main.tex:342 both state explicitly).


---

## 3.proofread — 26 candidates, NOT adjudicated

Lens: proofread. Confirmations reported by the lens: Scope: proofreading/submission-hygiene pass over texs/ai4dd/main.tex only (no edits made, per instructions). Anonymity and build cleanliness were not re-checked, per the task brief (already verified clean).

Checked and found clean / no issue:
- Straight quotes: none in rendered text; the paper's `` '' quote pairs (e.g. line 429 ``sealed'') are correctly typeset throughout.
- \% usage: consistent, no spelled-out "percent" anywhere.
- $\times$ vs "x": consistently `\times` in math mode (5\times4, 2\times2, 2\times2\times2), no stray "5x4" etc. in rendered text (only in the internal dev-comment block, which doesn't render).
- $\pm$ usage: consistent math-mode use for numeric values; the one pr

### [major] Appendix table breaks its own caption promise: no std devs, and a placeholder mismatch

- **Where:** 482-483, 490-496 (prose); figures/tolerance_table.tex lines 5 and 9
- **As written:** \Cref{tab:tolerance} repeats the tolerance sweep of \Cref{tab:main-results} with the fold-level standard deviation on every cell.
- **Problem:** The generated table that \Cref{tab:tolerance} pulls in via \input{figures/tolerance_table.tex} (lines 488-496) contains only bare point-estimate F1 values in every cell (e.g. row 5: "$0.573$ & $0.540$ & $0.461$ & $0.345$ & $+0.000$") -- no \pm std anywhere, directly contradicting the sentence that introduces it. The same row also prints "$+0.000$" for the base model's own growth column, where the parallel cell in \Cref{tab:main-results} (line 286) uses "---" for the identical self-comparison, so the two tables disagree on how to denote "not applicable."
- **Correction:** Either regenerate tolerance_table.tex to include the promised per-cell standard deviations, or rewrite the appendix sentence to describe what the table actually shows (point estimates only); and make the base row's placeholder consistent with "---" as used in Table 2.
- **Evidence:** Read figures/tolerance_table.tex directly: all 5 data rows have exactly 5 bare numbers plus one signed growth number, no \pm anywhere in the file.

### [major] "Kept their sign" example actually flips sign

- **Where:** 429
- **As written:** Effects that kept their sign still moved by a lot: the isolated gated adapter measured $-0.001$ on one fold and $+0.045$ on the other.
- **Problem:** This sentence is offered as the counterpart to the preceding sign-flipping examples (ESM-C swap, ESM-C 6B vs 600M, 3Di), i.e. as an effect whose sign did NOT change between the two held-out folds. But $-0.001$ and $+0.045$ are of opposite sign, so this is itself a sign flip and belongs with the examples in the sentence before it, not against them.
- **Correction:** Either replace with a genuine same-sign, large-swing example, or move this sentence to the sign-flipping list and rewrite the "kept their sign" claim with a different modification.
- **Evidence:** -0.001 < 0 and +0.045 > 0: opposite signs, the opposite of what "kept their sign" claims.

### [major] "Confident +0.03" contradicts the table and the per-fold numbers it cites

- **Where:** 440
- **As written:** nested cross-validation puts ESM-C 6B at $0.588\pm0.016$ against ESM-2's $0.573\pm0.025$ (\Cref{tab:main-results}), overlapping intervals, where this table records a confident $+0.03$.
- **Problem:** "This table" is Table \ref{tab:verdict}, whose own row for this exact modification (line 451) reads "$+0.03$, unstable across folds" -- the opposite of "confident." The instability is spelled out two paragraphs earlier at line 429, where the same embedding swap measures $+0.074$ on one held-out fold and $-0.011$ on the other (a sign flip).
- **Correction:** Drop "confident"; e.g. "...where this table records an unstable $+0.03$, one of the effects this appendix's opening argues cannot be trusted from a single split."
- **Evidence:** Table \ref{tab:verdict} row (line 451): "ESM-C 6B instead of ESM-2 & embedding swap & $+0.03$, unstable across folds & helps"; line 429: "+0.074 on one and -0.011 on the other."

### [major] "Five times wider" contradicts the paper's own embedding widths

- **Where:** 305
- **As written:** The intervals overlap, so an embedding five times wider buys no confirmed improvement on its own
- **Problem:** "Width" is the paper's own vocabulary for embedding dimension (line 212: "reduces its width (1280 to 256 for ESM-2)"). ESM-C 6B's width is given elsewhere in the paper as 2560 (line 455: "ESM-C 6B compression 2560$\to$256"). 2560/1280 = 2, not 5. Read as parameter count instead (650M at line 210 vs. 6B), the ratio is about 9, not 5 either. Neither reading inside the paper supports "five times."
- **Correction:** Say what is actually meant -- "twice as wide" (embedding dimension) or "roughly an order of magnitude more parameters" -- consistent with 1280/2560 and 650M/6B stated elsewhere in the paper.
- **Evidence:** Line 212: "1280 to 256 for ESM-2"; line 455: "2560$\to$256"; line 210: "a frozen pLM of 650 million." 2560/1280=2; 6e9/6.5e8≈9.2.

### [major] Headline gains don't match subtracting the table's own printed numbers

- **Where:** 148, 186, 297, 301, 305, 310 (prose); 286-292 (table)
- **As written:** combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base
- **Problem:** Table \ref{tab:main-results}'s own printed F1 values give $0.630-0.573=0.057$, not the $+0.058$ repeated four times in the paper (abstract, contributions, results, conclusion). The same pattern recurs elsewhere: line 301's exact-match gap "$+0.086$" vs. the table's $0.432-0.345=0.087$; line 305's "$+0.025$" vs. $0.371-0.345=0.026$; line 297's "recall rises by $0.042$" vs. the table's $0.579-0.538=0.041$. Most of these resolve if computed from the unrounded per-fold means rather than the rounded, printed 3-decimal cells (I verified this against runs/*/nested_cv_tolerance.json: e.g. the true combined gap is 0.05767, which rounds to 0.058) -- but the recall figure does not: the unrounded recall gap is 0.041463, which still rounds to 0.041, not 0.042. A reviewer checking the paper's own printed table will hit this repeatedly.
- **Correction:** Add a note to \Cref{tab:main-results}'s caption that reported deltas are computed from unrounded per-fold means and can differ from subtracting the printed (rounded) columns by up to 0.001; separately re-derive/re-check the "$+0.042$" recall figure at line 297 specifically, since it does not resolve even from full precision.
- **Evidence:** runs/5cv_baseline_esm2/nested_cv_tolerance.json and runs/5cv_esm2_full/nested_cv_tolerance.json cv_tol3_all_f1_mean = 0.572523 and 0.630193 (diff 0.05767->0.058, matches text); cv_tol3_all_recall_mean = 0.5377 (base) and 0.579163 (adapter), diff 0.041463->0.041, does not match text's 0.042.

### [minor] Missing object pronoun: "treating as a black box"

- **Where:** 340
- **As written:** DeepPeptide's architecture is worth describing in detail rather than treating as a black box.
- **Problem:** "Treating" has no object. Contrast the parallel construction four paragraphs later (line 344), which gets this right: "rather than treating it as a system to audit."
- **Correction:** "...rather than treating it as a black box."
- **Evidence:** Line 344 in the same appendix subsection: "rather than treating it as a system to audit" -- shows the intended pattern with the pronoun present.

### [minor] Missing connector: "segments (start, end) pairs"

- **Where:** 383
- **As written:** and contiguous runs of the same label form typed \textbf{segments} (start, end) pairs:
- **Problem:** Two noun phrases ("segments" and "(start, end) pairs") are juxtaposed with no connecting word or punctuation, making the clause ungrammatical/hard to parse. The main text's version of the same sentence (line 221) is simpler and doesn't have this problem: "contiguous runs of a label form typed segments."
- **Correction:** "...form typed \textbf{segments}: (start, end) pairs" or "...form typed \textbf{segments}, i.e.\ (start, end) pairs".
- **Evidence:** Compare the simpler, grammatical line 221: "contiguous runs of a label form typed segments."

### [minor] LoRA used unexpanded in the main text, defined only in the appendix

- **Where:** 314 (used); 472 (defined)
- **As written:** including a structural channel, an auxiliary bond loss, a telescopic segment CRF and LoRA fine-tuning, was only ever tested under the weaker protocol
- **Problem:** "LoRA" appears in the Limitations paragraph of the 5-page main text with no expansion. It is only spelled out later, in Appendix E: "Low-rank adaptation of the last few pLM layers" (line 472). A reader of the main text alone never sees the acronym defined.
- **Correction:** Expand on first use in the main text: "LoRA (low-rank adaptation) fine-tuning", or move the appendix's parenthetical gloss forward.
- **Evidence:** grep of main.tex shows "LoRA" first appears at line 314; "Low-rank adaptation" (the expansion) first appears at line 472, after the main text/References boundary.

### [minor] "Little loss from applying them together" describes a small gain, not a loss

- **Where:** 148
- **As written:** combine to a $+0.058$ F1 gain over the $0.573\pm0.025$ base, with little loss from applying them together
- **Problem:** The individual gains are $+0.024$ and $+0.029$ (summing to $+0.053$); the combined $+0.058$ exceeds that sum, i.e. the two additions are mildly super-additive together, not sub-additive. "Little loss" claims the opposite of what the numbers show. The Results section's own phrasing is more careful and doesn't have this problem: "together they give $+0.058$, close to the sum of the parts" (line 297).
- **Correction:** Drop "with little loss from applying them together", or replace with something like "with no loss from applying them together" / "slightly more than additive."
- **Evidence:** 0.024+0.029=0.053 < 0.058 (line 186/289); contrast with line 297's own "close to the sum of the parts."

### [minor] "Less than half" is false for the adapter's individual gain

- **Where:** 310
- **As written:** improve segment F1 by $+0.058$ together and by less than half that on their own
- **Problem:** Half of $+0.058$ is $+0.029$. The adapter's own reported individual gain is exactly $+0.029$ (lines 186, 289) -- not less than half. Using the underlying unrounded per-fold means (verified against runs/5cv_esm2_adapter_only and runs/5cv_baseline_esm2), the adapter's true gain is 0.029284, while half of the true combined gain (0.057670) is 0.028835 -- so the adapter's individual gain is actually slightly *more* than half, not less.
- **Correction:** "...and by roughly half that or less on their own", or single out the boundary head only (+0.024, which is comfortably under half).
- **Evidence:** runs/5cv_esm2_adapter_only vs runs/5cv_baseline_esm2 nested_cv_tolerance.json cv_tol3_all_f1_mean diff = 0.029284; runs/5cv_esm2_full diff = 0.057670; 0.029284 > 0.057670/2 = 0.028835.

### [minor] Oxford comma used inconsistently throughout

- **Where:** 148, 174, 178, 210, 214, 340, 395, 505
- **As written:** proteomics, vaccine design, and disease research
- **Problem:** The paper switches between an Oxford comma and none for three-item lists with no discernible pattern: WITH comma at line 148 ("proteomics, vaccine design, and disease research"), line 210 ("a segment start, an interior position, or a segment end"), line 395 ("segment length, type, and genus distributions"); WITHOUT at line 174 ("neurodegenerative, oncological and endocrine conditions"), line 178 ("a segment start, interior or end"), line 214 ("Code, the eight run configurations and a synthetic smoke test"), line 340 ("a convolution, a single-layer bidirectional LSTM and a second convolution"), line 505 ("F1, precision and recall").
- **Correction:** Pick one convention (Oxford comma is the more common ML-paper default) and apply it throughout.
- **Evidence:** Direct textual contrast between the listed lines; several are near-identical three-item lists.

### [minor] Comma inconsistently dropped after a fronted phrase

- **Where:** 174, 212, 303, 305, 415
- **As written:** In vaccine design the same prediction is useful at the design stage
- **Problem:** A comma is missing after the introductory phrase, unlike the very next sentence in the same paragraph: "In disease research, proteolysis is disrupted..." (line 174). The same slip recurs at line 212 ("Before the per-residue embedding reaches the CNN--BiLSTM it passes through..."), line 303 ("To hold detection fixed we compared..."), line 305 ("Under the earlier single-split protocol the boundary head on ESM-C 6B was worth..." -- contrast the correctly-punctuated parallel case at line 260, "Under an earlier single-split protocol we ran during development, that same modification..."), and line 415 ("Across the nested-CV grid the correction raises recall by...").
- **Correction:** Add a comma after each fronted phrase: "In vaccine design, the same prediction..."; "Before the per-residue embedding reaches the CNN--BiLSTM, it passes through..."; "To hold detection fixed, we compared..."; "Under the earlier single-split protocol, the boundary head..."; "Across the nested-CV grid, the correction raises recall...".
- **Evidence:** Line 174 has both the unpunctuated and (in the next sentence) correctly punctuated version of the identical construction side by side; line 260 vs. 305 is the same minimal pair.

### [minor] "Interior" vs "inside" used interchangeably for the same CRF-state concept

- **Where:** 206, 470
- **As written:** the boundary head produces a start/inside/end correction per type on top of it, so the first and last states of a segment no longer see the same evidence as its interior
- **Problem:** The same sentence uses both "inside" and "interior" for the identical concept (the middle CRF state of a segment). Every other occurrence of this term in the paper is "interior" (lines 148, 176 x2, 178, 201, 210, 340), making "inside" here and at line 470 ("a bias on the start, inside and end emissions specifically") an outlier.
- **Correction:** Use "interior" consistently, including in the Fig. 2 caption's "start/inside/end" label and at line 470.
- **Evidence:** grep count: "interior" used as this term 8 times across lines 148/176/176/178/201/210/340; "inside" used as this term only at 206 and 470.

### [minor] The label set $\{\textsf{None},\textsf{Peptide},\textsf{Propeptide}\}$ is typeset three different ways

- **Where:** 221, 340, 382
- **As written:** $y_t \in \{\textsf{None},\textsf{Peptide},\textsf{Propeptide}\}$
- **Problem:** The identical three-element set is spaced three different ways across the paper: no space after the commas at line 221; a normal LaTeX space at line 340 ("$\{\textsf{None}, \textsf{Peptide}, \textsf{Propeptide}\}$"); an explicit \, tie-space at line 382 ("$\{\textsf{None},\ \textsf{Peptide},\ \textsf{Propeptide}\}$").
- **Correction:** Use one spacing convention (a plain space after each comma is simplest) for all three occurrences.
- **Evidence:** Direct textual comparison of the three math-mode expressions at lines 221, 340, 382.

### [minor] Genus names mix \emph and \textit in the same caption

- **Where:** 244
- **As written:** \emph{Cyriopagopus} and \emph{Lycosa} are spider venoms, \emph{Conus} a cone snail; each is a homology cluster that GraphPart is obliged to keep intact. \textit{Homo}, by contrast, is spread evenly (71/81/66/130/84).
- **Problem:** Three genus names in this caption use \emph{} and the fourth (\textit{Homo}) uses \textit{} for the same purpose (italicizing a genus name), with no apparent reason for the switch.
- **Correction:** Use \emph{Homo} for consistency with the other three genus names in the same sentence.
- **Evidence:** Direct textual comparison within \caption of Table \ref{tab:genus}, line 244.

### [minor] "Labelled" is the sole British spelling in an otherwise American-English document

- **Where:** 221
- **As written:** A precursor $x=(a_1,\dots,a_L)$ is labelled per residue with $y_t \in \{\textsf{None},\textsf{Peptide},\textsf{Propeptide}\}$
- **Problem:** The paper consistently uses American -ize/-ization spellings elsewhere (initialized: 148, 178, 184, 210, 310; localization: 159, 178, 212, 303, 483; optimized: 340; standardized/summarized: 342, 477), making the British double-l "labelled" here the only outlier.
- **Correction:** "labeled" (single l), for consistency with the rest of the paper's American spelling.
- **Evidence:** grep of -ize/-ized/-ization forms across main.tex shows only American spellings elsewhere in the rendered (non-comment) text.

### [minor] Thousands separators used in body text but dropped in tables for the same quantities

- **Where:** 224 (text); 359-362, 251-256 (tables)
- **As written:** The collection grows from 8{,}449 proteins to 9{,}619, of which 1{,}178 were absent in 2022
- **Problem:** Body text consistently uses "{,}" thousands separators for counts of this kind (also 4{,}678, 235{,}113, 331{,}008, 1{,}600, 2{,}300, 26{,}730, 29{,}051, 29{,}132, 2{,}322 elsewhere), but Table \ref{tab:dataset} (lines 359-362: "8449", "9619", "1178", "6372", "7431", "8211", "9140") and Table \ref{tab:genus} (lines 251-256: "8897", "1558", "2572", "1263", "2025", "1479", "714", "293", "163") print the identical kind of protein/segment counts with no separators at all.
- **Correction:** Add "{,}" thousands separators to the table entries, or drop them from the body text, so the convention matches across running text and tables.
- **Evidence:** Direct comparison of line 224's "8{,}449"/"9{,}619" against line 359's "8449"/"9619" for the identical numbers.

### [minor] "20 cells" vs "twenty cells" for the identical quantity

- **Where:** 238, 262, 486
- **As written:** so results below pool them (model-select $\cup$ sealed, $\approx$2{,}300 proteins) to reduce
- **Problem:** This is a formatting note, not the quote's own content: the number of cells per nested-CV configuration is written as the numeral "20" at lines 238 ("not across all 20 cells") and 262 ("a configuration of 20 cells"), but spelled out as "twenty" at line 486 ("pooled over the twenty cells of each configuration").
- **Correction:** Use "20 cells" throughout for consistency.
- **Evidence:** grep shows "20 cells" at lines 238 and 262, "twenty cells" at line 486, referring to the same fixed protocol quantity (5 outer x 4 inner).

### [minor] "Nested-cross-validation" hyphenated once, unhyphenated everywhere else

- **Where:** 417
- **As written:** and the full $5\times4$ nested-cross-validation grid in \Cref{tab:main-results}
- **Problem:** Every other occurrence of this term in the paper (15 instances, e.g. lines 148, 180, 238, 271, 342, 425, 440, 463, 490) writes "nested cross-validation" without a hyphen between "nested" and "cross-validation", including in equally attributive uses (e.g. line 271's caption "$5\times4$ nested cross-validation with the corrected matcher").
- **Correction:** Drop the hyphen: "nested cross-validation grid", for consistency with every other instance.
- **Evidence:** grep -o 'nested[- ]cross-validation' main.tex returns 16 matches, only line 417 hyphenated.

### [nit] "Swiss-Prot" vs "UniProtKB/Swiss-Prot" naming

- **Where:** 224
- **As written:** DeepPeptide built its dataset from \texttt{PEPTIDE} and \texttt{PROPEP} annotations in the 2022 Swiss-Prot release.
- **Problem:** Four other mentions of this database use the fuller "UniProtKB/Swiss-Prot" (lines 148, 180, 342 x2); this is the only place the shorter "Swiss-Prot" alone is used for the same database.
- **Correction:** "...in the 2022 UniProtKB/Swiss-Prot release", for consistency with the rest of the paper.
- **Evidence:** grep -o 'UniProtKB/Swiss-Prot' main.tex: 4 matches (148, 180, 342x2); grep -o 'Swiss-Prot' main.tex: 5 matches, the extra one at 224.

### [nit] Comma splice

- **Where:** 201
- **As written:** the decoder is boundary-aware, the features feeding it are not
- **Problem:** Two independent clauses ("the decoder is boundary-aware" / "the features feeding it are not [boundary-aware]") are joined only by a comma, with no conjunction. May be an intentional terse antithesis, but the task brief flags comma splices as a characteristic non-native slip worth checking.
- **Correction:** "...the decoder is boundary-aware; the features feeding it are not." (semicolon), or "...boundary-aware -- the features feeding it are not."
- **Evidence:** Direct reading of the clause; no coordinating conjunction present between the two independent clauses.

### [nit] Bare decimals outside math mode, inconsistent with the rest of the paper

- **Where:** 233
- **As written:** DeepPeptide reports precision 0.68 and recall 0.49 at $\pm3$ under nested cross-validation on the 2022 data, an implied F1 of 0.570.
- **Problem:** "0.68", "0.49" and "0.570" are set as plain text, not in math mode ($...$), unlike essentially every other decimal figure in the paper, including "$0.573\pm0.025$" later in the very same sentence.
- **Correction:** "$0.68$" / "$0.49$" / "$0.570$" to match the paper's math-mode convention for numbers.
- **Evidence:** Same sentence contains both the unwrapped "0.68"/"0.49"/"0.570" and the math-mode "$0.573\pm0.025$".

### [nit] Table \ref{tab:verdict} mixes 2- and 3-decimal precision in one column

- **Where:** 451-452, 456
- **As written:** Boundary head on ESM-C 6B & decoder addition & $+0.05$, $+0.07$, consistent & helps
- **Problem:** The "Measured effect on F1" column reports some rows to 2 decimal places ("$+0.03$" line 451, "$+0.05$, $+0.07$" line 452) and others to 3 ("$+0.022$", line 456), within the same column of the same table. The "+0.05, +0.07" figures also appear with a third decimal in prose at line 305 ("$+0.053$ and $+0.067$") for the same numbers.
- **Correction:** Report all rows in this column to the same number of decimal places (3, matching the main results table).
- **Evidence:** Direct comparison of lines 451/452 vs 456 within the same table column; cross-reference to line 305's 3-decimal version of the same boundary-head-on-ESM-C-6B numbers.

### [nit] Precursor sequence $x$ notated two different ways

- **Where:** 221, 380
- **As written:** x = (a_1, a_2, \dots, a_L), \qquad a_t \in \Omega,
- **Problem:** The same object is defined twice with different notation: line 221 gives "$x=(a_1,\dots,a_L)$" (no spaces around "=", $a_2$ omitted), while line 380's formal definition spaces the "=" and shows $a_2$ explicitly.
- **Correction:** Use one consistent form for $x$'s definition in both places.
- **Evidence:** Direct textual comparison of the math expressions at lines 221 and 380.

### [nit] En dash used where em dash is conventional for parenthetical asides

- **Where:** 176, 340, 381, 391, 463
- **As written:** the position-specific evidence a well-designed decoder could condition on is simply never computed by the encoder that feeds it
- **Problem:** The paper consistently renders parenthetical interruptions with " -- " (LaTeX en dash, --) rather than " --- " (em dash), e.g. also at lines 340, 381, 391, 463. This is a legitimate house-style choice (spaced en dash for parentheticals, as in some journals) and is applied consistently, so flagging mainly for the authors to confirm it's deliberate rather than a LaTeX slip (single "--" typed where "---" was intended).
- **Correction:** If em dash is intended, change "--" to "---" at the listed lines; otherwise no change needed since usage is internally consistent.
- **Evidence:** grep -n ' -- ' main.tex: all 5 rendered-text instances use exactly two hyphens, none use three.

### [nit] \Cref not preceded by a tie (~) after a plain word space

- **Where:** 224, 260, 299, 303, 391, 415
- **As written:** Segment-length filtering, motif balancing and the full composition of the rebuild are given in \Cref{sec:app-problem}.
- **Problem:** Throughout the paper, \Cref is preceded by a breakable word-space rather than a non-breaking tie (~) whenever it follows a word directly (as opposed to an open parenthesis, where no space intervenes) -- e.g. also lines 260, 299 (twice), 303, 391, 415. A line break could in principle strand the cross-reference number at the start of a line. cleveref's own internal glue mitigates this somewhat, so this is cosmetic.
- **Correction:** Insert ~ before \Cref in these word-space cases: "given in~\Cref{sec:app-problem}", etc.
- **Evidence:** grep -noE '.{15}\\Cref' main.tex shows zero instances of "~\Cref" anywhere in the file.


---

## 4. Metric-suite findings (separate topic)

The earlier version of this file also carried 28 findings about
`analysis/metrics/src/segment_quality_cv.py`, which is code, not the paper. Three of them
were confirmed and are now fixed in commit `1b2dc03` (per-end interior filter, residue MCC,
per-fold sign convention). The full list, including the ones never adjudicated, is preserved
in the previous revision of this file:

```
git show 141623b:texs/ai4dd/PENDING_REVIEW.md
```

Still unchecked there: whether the length-bin fix and the protein-set intersection hold up.

---

## 5. Full precision audit of every run (2026-09-04)

`amp=True` means bf16 autocast during training: `src/train_loop_crf.py:504`,
`use_amp = getattr(args, "amp", False)`, so `amp_dtype` is inert when `amp` is false.
The paper never mentions training precision anywhere.

`amp=True` marks the OLDER runs — the project switched to fp32 part-way through, and the
switch cut across several comparisons that the paper presents as controlled.

| artefact | bf16 side | fp32 side | verdict |
|---|---|---|---|
| Tables 1-2, nested CV | `5cv_baseline_esm2` (all 20 cells) | the four variants (80 cells) | **confounded** |
| Fig. 5 scoreboard, Fig. 7 tolerance | `2026_baseline_esm2`, `2026_esmc_600m` | `2026_esmc_6b` and every addition | **mixed** |
| ESM-C 6B vs ESM-C 600M (+0.028 / −0.021) | 600M | 6B | **confounded** |
| ESM-C 6B instead of ESM-2, +0.03 | ESM-2 | ESM-C 6B | **confounded** |
| boundary head on ESM-2, single split (≈0) | baseline | variant | **confounded** |
| boundary head on ESM-C 6B (+0.05 / +0.07) | — | both | clean |
| gated adapter +0.022 | — | both | clean |
| 3Di channel, bond loss | — | both | clean |
| Figs. 10-11, data scaling | whole ESM-2 curve (`scale_baseline_*` + `2026_baseline_esm2`) | whole ESM-C curves (`scale_proj_*`, `scale_3di_*`) | within-curve claims clean, between-curve level **confounded** |
| LoRA vs frozen baseline (appendix D) | both (`esm2_lora_*`, `train_run_esm2`) | — | clean |

### A free estimate of how large the confound is

`runs/train_run_esm2` and `runs/train_run_esm2_100` are the same ESM-2 baseline on the same
2022 GraphPart split, same architecture (`LSTMCNN`), same epochs (100), lr (1e-4) and batch
size (48). The only substantive config difference is `amp: True` vs `amp: False`; every other
differing key is `None` against a default the newer config schema records explicitly.

```
train_run_esm2      (bf16)  test f1 all = 0.6073
train_run_esm2_100  (fp32)  test f1 all = 0.5881
```

bf16 scored **0.019 higher**. If that direction carries over, the paper's effects are
*understated*, not inflated: every variant is fp32 and is being compared against a baseline
that bf16 was helping. But `seed` is `None` in both configs, so this is one seed against one
seed, and 0.019 is the same size as the effects under discussion (+0.024 head, +0.029 adapter,
+0.058 combined). It bounds the confound at roughly the size of what is being measured, which
is exactly why it cannot be waved away.

The clean fix is one fp32 re-run of `5cv_baseline_esm2`, which is GPU work. The cheap fix is
to state the difference and this estimate in the paper.
