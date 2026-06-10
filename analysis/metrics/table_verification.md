# Отчёт о сверке таблиц

> Исторический верификационный лог: сверка значений в `texs/Overleaf/experiments.tex` с
> JSON-метриками запусков. Большинство находок уже исправлено (см.
> `texs/error_analysis/reproducibility_and_na_metrics.md`); каноническая сводка —
> `canonical_metrics.md` и `big_metrics_table.md`. Таблицы с символами ✓/Δ оставлены
> как есть (язык-нейтральны).

Источник: `texs/Overleaf/experiments.tex`  
Каталог запусков: `runs/*/`  
Допуск совпадения (P/R/F1): 1e-4  
Порог «устаревших» MCC/AUC: 1e-4

## Резюме

- Всего строк в 3 таблицах: 47
- Чисто совпали (P/R/F1 точно + MCC/AUC в пределах 1e-4): 40
- Совпали P/R/F1, но устаревшие только MCC/AUC (Δ > 1e-4): 4
- Ошибка переписывания пропептидов (all+pep точно, Pprop/Rprop неверно): 1
- Не совпали по P/R/F1 (нет исходного запуска): 2
- Папок запусков, не упомянутых ни в одной таблице: 14

## Таблица 1 (Архитектурные изменения, все данные)

| Row label | Matched folder | P/R/F1 source | F1-all match? | MCC/AUC match? (max Δ) | Notes |
|-----------|---------------|---------------|---------------|------------------------|-------|
| `ESM2 (baseline)` | `train_run_esm2` | `test_metrics.json` | ✓ exact | ✗ STALE (Δ=0.0004) |  |
| `ESM2 + telescopic CRF` | `esm2_telescoping_segmental` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 + Aho emission fusion` | `esm2_aho_emission_fusion` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 + (Aho -> hidden layer 32) emission fusion` | `esm2_aho_emission_fusion_h32` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 + Aho hidden state fusion` | `esm2_aho_mid_fusion_raw_m64` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 + Aho hidden state fusion only peptides` | `esm2_aho_mid_fusion_raw_m64_pep_only` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 + Aho сигнал добавляется к CRF переходам` | `esm2_aho_transition_bias_sparse_trainable_zero` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 + Aho early fusion (concat with esm)` | `esm2_aho_tribranch` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 + доп. лосс разрезов к ближайшей границе` | `esm2_bond_loss_soft_l005_w5_tau15` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2 c AdamW оптимизатором` | `train_run_esm2_adamw` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |

## Таблица 2 (Генераторы эмбеддингов, все данные)

| Row label | Matched folder | P/R/F1 source | F1-all match? | MCC/AUC match? (max Δ) | Notes |
|-----------|---------------|---------------|---------------|------------------------|-------|
| `ESM2` | `train_run_esm2` | `test_metrics.json` | ✓ exact | ✗ STALE (Δ=0.0004) |  |
| `ESM2+residue features (ESM2+ below)` | `train_run_esm2_plus` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM-C` | `train_run_esmc_600m` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM-C 6B` | `esmc_6b` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ProstT5` | `train_run_prostt5` | `test_metrics.json` | ✓ exact | ✗ STALE (Δ=0.0006) |  |
| `ProstT5+residue features` | `train_run_prostt5_plus` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2) proj.` | `train_run_esm2+3di_proj` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2) proj.gated.` | `train_run_esm2+3di_proj_gated` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2) proj.gated.conv.` | `train_run_esm2+3di_proj_gated_conv` | `test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2+) proj.gated.conv.` | `**UNMATCHED**` | `test_metrics.json` | ✗ (Δ=0.0195) | — (no matched folder) | Closest: train_run_esm2 (diff=0.019495) |
| `AFTK all, no filter` | `train_run_aft` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `AFTK only single, no filter` | `train_run_aft_single` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `AFTK all w/o lddt, no filter` | `train_run_aft_no_lddt` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `AFTK all, >70\% avg plddt` | `train_run_aft_plddt70` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK all, no filter) pr.gt.conv` | `train_run_esm2_aft` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK only single no filter) pr.gt.conv` | `train_run_esm2_aft_single_gated` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK only pair no filter) pr.gt.conv` | `train_run_esm2_aft_pair_gated` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK all w/o lddt no filter) pr.gt.conv` | `train_run_esm2_aft_no_lddt_gated` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK all, >70\% avg plddt) pr.gt.conv` | `train_run_esm2_aft_plddt70` | `test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |

## Таблица 3 (Только Homo)

| Row label | Matched folder | P/R/F1 source | F1-all match? | MCC/AUC match? (max Δ) | Notes |
|-----------|---------------|---------------|---------------|------------------------|-------|
| `ESM2` | `train_run_esm2` | `homo_test_metrics.json` | ✓ exact | ✗ STALE (Δ=0.0044) | PARTIAL MATCH: all+pep exact, propeptides mismatch (Pprop: table=0.665728 json=0.551020, Rprop: table=0.511913 json=0.50 |
| `ESM2+residue features (ESM2+ below)` | `train_run_esm2_plus` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM-C` | `train_run_esmc_600m` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ProstT5` | `train_run_prostt5` | `homo_test_metrics.json` | ✓ exact | ✗ STALE (Δ=0.0133) |  |
| `ProstT5+residue features` | `train_run_prostt5_plus` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2) proj.` | `train_run_esm2+3di_proj` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2) proj. gated.` | `train_run_esm2+3di_proj_gated` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2) proj.gated.conv.` | `train_run_esm2+3di_proj_gated_conv` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `(ProstT5 3DI + ESM2+) proj.gated.conv.` | `**UNMATCHED**` | `homo_test_metrics.json` | ✗ (Δ=0.0233) | — (no matched folder) | Closest: esm2_aho_emission_fusion (diff=0.023256) |
| `AFTK all, no filter` | `train_run_aft` | `homo_test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `AFTK only single, no filter` | `train_run_aft_single` | `homo_test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `AFTK all w/o lddt, no filter` | `train_run_aft_no_lddt` | `homo_test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `AFTK all, >70\% avg plddt` | `train_run_aft_plddt70` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK all, no filter) pr.gt.conv` | `train_run_esm2_aft` | `homo_test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK only single, no filter) pr.gt.conv` | `train_run_esm2_aft_single_gated` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK only pair, no filter) pr.gt.conv` | `train_run_esm2_aft_pair_gated` | `homo_test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK all w/o lddt, no filter) pr.gt.conv` | `train_run_esm2_aft_no_lddt_gated` | `homo_test_metrics_infer.json` | ✓ exact | ✓ (0.00e+00) |  |
| `ESM2+(AFTK all, >70\% avg plddt) pr.gt.conv` | `train_run_esm2_aft_plddt70` | `homo_test_metrics.json` | ✓ exact | ✓ (0.00e+00) |  |

## НАХОДКИ

Всего расхождений: 8

### Finding 1: [HIGH] PARTIAL_MATCH_PROPEPTIDES_ERROR
- **Table:** Table 3 (Homo only)
- **Row:** `ESM2`
- **Matched folder:** `train_run_esm2`
- **Max delta:** 1.1471e-01
- **Detail:** PARTIAL MATCH: all+pep exact, propeptides mismatch (Pprop: table=0.665728 json=0.551020, Rprop: table=0.511913 json=0.509434). Full P/R/F1 diff=1.1471e-01. Only Pprop and Rprop are corrupted; F1prop is correct. The erroneous values are exact 16-digit copies of `train_run_esm2_plus/test_metrics.json` propeptides P/R.

### Finding 2: [HIGH] UNMATCHED
- **Table:** Table 3 (Homo only)
- **Row:** `(ProstT5 3DI + ESM2+) proj.gated.conv.`
- **Matched folder:** `esm2_aho_emission_fusion`
- **Max delta:** 2.3256e-02
- **Detail:** P/R/F1 max diff to nearest run = 0.023256. No matching run folder found. `train_run_esm2_plus_proj_gated` is the closest-named folder in runs/ but its P/R/F1 values do NOT match (max diff ≈0.02), so it is not the source; the proj.gated.conv. ESM2+ run is absent from runs/.

### Finding 3: [HIGH] UNMATCHED
- **Table:** Table 2 (Embedding generators, all data)
- **Row:** `(ProstT5 3DI + ESM2+) proj.gated.conv.`
- **Matched folder:** `train_run_esm2`
- **Max delta:** 1.9495e-02
- **Detail:** P/R/F1 max diff to nearest run = 0.019495. No matching run folder found. `train_run_esm2_plus_proj_gated` is the closest-named folder in runs/ but its P/R/F1 values do NOT match (max diff ≈0.02), so it is not the source; the proj.gated.conv. ESM2+ run is absent from runs/.

### Finding 4: [MEDIUM] STALE_MCC_AUC
- **Table:** Table 3 (Homo only)
- **Row:** `ProstT5`
- **Matched folder:** `train_run_prostt5`
- **Max delta:** 1.3284e-02
- **Detail:** MCC-all: table=0.701648 json=0.707284 Δ=5.6354e-03; AUC-all: table=0.698877 json=0.694519 Δ=4.3584e-03; MCC-pep: table=0.574985 json=0.588269 Δ=1.3284e-02; AUC-pep: table=0.856257 json=0.850662 Δ=5.5952e-03

### Finding 5: [MEDIUM] STALE_MCC_AUC
- **Table:** Table 3 (Homo only)
- **Row:** `ESM2`
- **Matched folder:** `train_run_esm2`
- **Max delta:** 4.3929e-03
- **Detail:** MCC-all: table=0.693992 json=0.695702 Δ=1.7108e-03; AUC-all: table=0.604698 json=0.605724 Δ=1.0261e-03; MCC-pep: table=0.568299 json=0.572692 Δ=4.3929e-03; AUC-pep: table=0.881829 json=0.882533 Δ=7.0422e-04

### Finding 6: [MEDIUM] STALE_MCC_AUC
- **Table:** Table 2 (Embedding generators, all data)
- **Row:** `ProstT5`
- **Matched folder:** `train_run_prostt5`
- **Max delta:** 5.6699e-04
- **Detail:** MCC-all: table=0.716861 json=0.717258 Δ=3.9618e-04; AUC-all: table=0.785958 json=0.786320 Δ=3.6181e-04; MCC-pep: table=0.588545 json=0.589112 Δ=5.6699e-04; AUC-pep: table=0.854901 json=0.855385 Δ=4.8345e-04

### Finding 7: [MEDIUM] STALE_MCC_AUC
- **Table:** Table 1 (Arch changes, all data)
- **Row:** `ESM2 (baseline)`
- **Matched folder:** `train_run_esm2`
- **Max delta:** 3.7997e-04
- **Detail:** AUC-all: table=0.716213 json=0.716044 Δ=1.6879e-04; MCC-pep: table=0.697816 json=0.697436 Δ=3.7997e-04; AUC-pep: table=0.858958 json=0.859269 Δ=3.1140e-04

### Finding 8: [MEDIUM] STALE_MCC_AUC
- **Table:** Table 2 (Embedding generators, all data)
- **Row:** `ESM2`
- **Matched folder:** `train_run_esm2`
- **Max delta:** 3.7997e-04
- **Detail:** AUC-all: table=0.716213 json=0.716044 Δ=1.6879e-04; MCC-pep: table=0.697816 json=0.697436 Δ=3.7997e-04; AUC-pep: table=0.858958 json=0.859269 Δ=3.1140e-04

### Переиспользование папок между таблицами (ожидаемо для базовой)
- `train_run_aft`: [('T2', 'AFTK all, no filter'), ('T3', 'AFTK all, no filter')]
- `train_run_aft_no_lddt`: [('T2', 'AFTK all w/o lddt, no filter'), ('T3', 'AFTK all w/o lddt, no filter')]
- `train_run_aft_plddt70`: [('T2', 'AFTK all, >70\\% avg plddt'), ('T3', 'AFTK all, >70\\% avg plddt')]
- `train_run_aft_single`: [('T2', 'AFTK only single, no filter'), ('T3', 'AFTK only single, no filter')]
- `train_run_esm2`: [('T1', 'ESM2 (baseline)'), ('T2', 'ESM2'), ('T3', 'ESM2')]
- `train_run_esm2+3di_proj`: [('T2', '(ProstT5 3DI + ESM2) proj.'), ('T3', '(ProstT5 3DI + ESM2) proj.')]
- `train_run_esm2+3di_proj_gated`: [('T2', '(ProstT5 3DI + ESM2) proj.gated.'), ('T3', '(ProstT5 3DI + ESM2) proj. gated.')]
- `train_run_esm2+3di_proj_gated_conv`: [('T2', '(ProstT5 3DI + ESM2) proj.gated.conv.'), ('T3', '(ProstT5 3DI + ESM2) proj.gated.conv.')]
- `train_run_esm2_aft`: [('T2', 'ESM2+(AFTK all, no filter) pr.gt.conv'), ('T3', 'ESM2+(AFTK all, no filter) pr.gt.conv')]
- `train_run_esm2_aft_no_lddt_gated`: [('T2', 'ESM2+(AFTK all w/o lddt no filter) pr.gt'), ('T3', 'ESM2+(AFTK all w/o lddt, no filter) pr.g')]
- `train_run_esm2_aft_pair_gated`: [('T2', 'ESM2+(AFTK only pair no filter) pr.gt.co'), ('T3', 'ESM2+(AFTK only pair, no filter) pr.gt.c')]
- `train_run_esm2_aft_plddt70`: [('T2', 'ESM2+(AFTK all, >70\\% avg plddt) pr.gt.c'), ('T3', 'ESM2+(AFTK all, >70\\% avg plddt) pr.gt.c')]
- `train_run_esm2_aft_single_gated`: [('T2', 'ESM2+(AFTK only single no filter) pr.gt.'), ('T3', 'ESM2+(AFTK only single, no filter) pr.gt')]
- `train_run_esm2_plus`: [('T2', 'ESM2+residue features (ESM2+ below)'), ('T3', 'ESM2+residue features (ESM2+ below)')]
- `train_run_esmc_600m`: [('T2', 'ESM-C'), ('T3', 'ESM-C')]
- `train_run_prostt5`: [('T2', 'ProstT5'), ('T3', 'ProstT5')]
- `train_run_prostt5_plus`: [('T2', 'ProstT5+residue features'), ('T3', 'ProstT5+residue features')]

### Папки запусков, не упомянутые ни в одной таблице

- `esm2_aho_state_bias_pep_boundary_010`
- `esm2_boundary_bond_l002_w5_tau15`
- `esm2_boundary_only_scale10`
- `esm2_lora_lstmcnncrf`
- `esm2_lora_lstmcnncrf_r4_last2_qv`
- `train_run_3di_only`
- `train_run_esm2_100`
- `train_run_esm2_25`
- `train_run_esm2_50`
- `train_run_esm2_75`
- `train_run_esm2_conv`
- `train_run_esm2_only_homo`
- `train_run_esm2_plus_proj_gated`
- `uni2026_run_esm2`

## Детальное сравнение MCC/AUC по строкам

### Таблица 1 (Архитектурные изменения, все данные)

| Row | Folder | MCC-all Δ | AUC-all Δ | MCC-pep Δ | AUC-pep Δ | MCC-prop Δ | AUC-prop Δ |
|-----|--------|-----------|-----------|-----------|-----------|------------|------------|
| `ESM2 (baseline)` | `train_run_esm2` | 9.65e-05 | 1.69e-04 ⚠ | 3.80e-04 ⚠ | 3.11e-04 ⚠ | 3.03e-05 | 3.99e-05 |
| `ESM2 + telescopic CRF` | `esm2_telescoping_segmental` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 + Aho emission fusion` | `esm2_aho_emission_fusion` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 + (Aho -> hidden layer 32) emission` | `esm2_aho_emission_fusion_h32` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 + Aho hidden state fusion` | `esm2_aho_mid_fusion_raw_m64` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 + Aho hidden state fusion only pept` | `esm2_aho_mid_fusion_raw_m64_pep_only` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 + Aho сигнал добавляется к CRF пере` | `esm2_aho_transition_bias_sparse_trainable_zero` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 + Aho early fusion (concat with esm` | `esm2_aho_tribranch` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 + доп. лосс разрезов к ближайшей гр` | `esm2_bond_loss_soft_l005_w5_tau15` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2 c AdamW оптимизатором` | `train_run_esm2_adamw` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

### Таблица 2 (Генераторы эмбеддингов, все данные)

| Row | Folder | MCC-all Δ | AUC-all Δ | MCC-pep Δ | AUC-pep Δ | MCC-prop Δ | AUC-prop Δ |
|-----|--------|-----------|-----------|-----------|-----------|------------|------------|
| `ESM2` | `train_run_esm2` | 9.65e-05 | 1.69e-04 ⚠ | 3.80e-04 ⚠ | 3.11e-04 ⚠ | 3.03e-05 | 3.99e-05 |
| `ESM2+residue features (ESM2+ below)` | `train_run_esm2_plus` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM-C` | `train_run_esmc_600m` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM-C 6B` | `esmc_6b` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ProstT5` | `train_run_prostt5` | 3.96e-04 ⚠ | 3.62e-04 ⚠ | 5.67e-04 ⚠ | 4.83e-04 ⚠ | 7.85e-05 | 2.80e-05 |
| `ProstT5+residue features` | `train_run_prostt5_plus` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2) proj.` | `train_run_esm2+3di_proj` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2) proj.gated.` | `train_run_esm2+3di_proj_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2) proj.gated.conv.` | `train_run_esm2+3di_proj_gated_conv` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2+) proj.gated.conv.` | train_run_esm2 | — | — | — | — | — | — |
| `AFTK all, no filter` | `train_run_aft` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `AFTK only single, no filter` | `train_run_aft_single` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `AFTK all w/o lddt, no filter` | `train_run_aft_no_lddt` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `AFTK all, >70\% avg plddt` | `train_run_aft_plddt70` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK all, no filter) pr.gt.conv` | `train_run_esm2_aft` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK only single no filter) pr.gt.` | `train_run_esm2_aft_single_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK only pair no filter) pr.gt.co` | `train_run_esm2_aft_pair_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK all w/o lddt no filter) pr.gt` | `train_run_esm2_aft_no_lddt_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK all, >70\% avg plddt) pr.gt.c` | `train_run_esm2_aft_plddt70` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

### Таблица 3 (Только Homo)

| Row | Folder | MCC-all Δ | AUC-all Δ | MCC-pep Δ | AUC-pep Δ | MCC-prop Δ | AUC-prop Δ |
|-----|--------|-----------|-----------|-----------|-----------|------------|------------|
| `ESM2` | `train_run_esm2` | 1.71e-03 ⚠ | 1.03e-03 ⚠ | 4.39e-03 ⚠ | 7.04e-04 ⚠ | 0.00e+00 | 0.00e+00 |
| `ESM2+residue features (ESM2+ below)` | `train_run_esm2_plus` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM-C` | `train_run_esmc_600m` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ProstT5` | `train_run_prostt5` | 5.64e-03 ⚠ | 4.36e-03 ⚠ | 1.33e-02 ⚠ | 5.60e-03 ⚠ | 0.00e+00 | 0.00e+00 |
| `ProstT5+residue features` | `train_run_prostt5_plus` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2) proj.` | `train_run_esm2+3di_proj` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2) proj. gated.` | `train_run_esm2+3di_proj_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2) proj.gated.conv.` | `train_run_esm2+3di_proj_gated_conv` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `(ProstT5 3DI + ESM2+) proj.gated.conv.` | esm2_aho_emission_fusion | — | — | — | — | — | — |
| `AFTK all, no filter` | `train_run_aft` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `AFTK only single, no filter` | `train_run_aft_single` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `AFTK all w/o lddt, no filter` | `train_run_aft_no_lddt` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `AFTK all, >70\% avg plddt` | `train_run_aft_plddt70` | 0.00e+00 | 0.00e+00 | nan | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK all, no filter) pr.gt.conv` | `train_run_esm2_aft` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK only single, no filter) pr.gt` | `train_run_esm2_aft_single_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK only pair, no filter) pr.gt.c` | `train_run_esm2_aft_pair_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK all w/o lddt, no filter) pr.g` | `train_run_esm2_aft_no_lddt_gated` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `ESM2+(AFTK all, >70\% avg plddt) pr.gt.c` | `train_run_esm2_aft_plddt70` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
