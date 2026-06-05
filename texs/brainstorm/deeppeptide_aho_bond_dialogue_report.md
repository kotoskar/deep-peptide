# Отчёт по ветке Aho-prior, fusion-экспериментам и идее bond loss

Дата: 2026-05-07  
Контекст: исследование по предсказанию сегментации белка на PEPTIDE / PROPEP на базе DeepPeptide-подхода.

---

## 1. Исходная постановка

Целевая задача: по последовательности precursor-белка предсказать сегменты:

- `PEPTIDE` - зрелые / биологически активные пептиды;
- `PROPEP` - пропептидные участки;
- фон / остальная последовательность.

Базовая архитектура - DeepPeptide-подобная модель:

```text
protein sequence
→ precomputed residue embeddings, например ESM2 [L, 1280]
→ DataLoader tensor [B, C, L]
→ LSTMCNN feature extractor
→ per-residue class emissions [B, L, 3]
→ repeat emissions into CRF states [B, L, 101]
→ constrained CRF loss / Viterbi decoding
→ predicted PEPTIDE / PROPEP segments
```

Ключевая особенность baseline: нейросеть выдаёт только coarse logits `None / Peptide / Propeptide`. Разделение на start / inside / end кодируется CRF state-space, а не отдельными нейросетевыми logits.

---

## 2. Исходные идеи

### Идея A: Aho-Corasick prior

Предложение: прогонять белок через Aho-Corasick со словарём известных биологически активных пептидов / пропептидов и давать модели знание о том, какие подотрезки уже известны.

Ожидаемая польза:

- модель получает сильный prior: “этот интервал уже известен как peptide-like”; 
- границы mature peptides могут определяться точнее;
- модель может выбирать из известных биологических вариантов, а не полностью “придумывать” сегменты.

Риск:

- модель может стать менее способной находить novel peptides;
- direct dictionary lookup может приводить к leakage, если test gold-пептиды попали во внешний словарь;
- Aho-сигнал может быть полезен для `PEPTIDE`, но не обязательно для `PROPEP`.

### Идея B: auxiliary bond / cleavage loss

Предложение: добавить auxiliary loss не по сегментам целиком, а по связям между соседними остатками:

```text
bond_i = есть ли разрез между residue i и i+1
```

Идея в развитом виде:

- использовать внешнюю cleavage-модель как дополнительный feature / prior;
- учить собственный auxiliary bond head на границах UniProt PEPTIDE / PROPEP;
- сделать loss distance-aware: если предсказанная граница рядом с настоящей, штраф должен быть меньше, чем если она далеко.

---

## 3. Aho-only: что реализовано

Были подготовлены унифицированные словари в `data/aho_train/`:

```text
uniprot_2022.tsv
dramp_general.tsv
dramp_natural.tsv
dbamp_3.tsv
apd6_natural.tsv
ampdb.tsv
```

Реализован `train_loop_aho.py`:

- загружает `labeled_sequences.csv` и `graphpart_assignments.csv`;
- для folded sources, например `uniprot_2022`, использует только нужные train-folds;
- external sources без fold-разметки загружает целиком;
- строит Aho-автомат;
- выбирает непересекающиеся интервалы через scoring;
- считает segment-level метрики;
- поддерживает strict-режим `drop_eval_exact`, где exact eval gold sequences удаляются из external dictionary.

Важный баг был найден и исправлен: изначально matching предсказанных и gold-сегментов не проверял `protein_id`. Это завышало F1, особенно для `PROPEP`. После фикса метрики стали protein-aware.

---

## 4. Aho-only: ключевые результаты

### 4.1 UniProt-only Aho

`uniprot_2022` train-fold dictionary почти не переносится на GraphPart test:

| Setup | Pep F1 | All F1 | Propep F1 | Вывод |
|---|---:|---:|---:|---|
| UniProt-only Aho | ~0.029 | ~0.013 | ~0.000 | Exact memorization train-сегментов почти не решает задачу |

Вывод: DeepPeptide / ESM2 не сводится к запоминанию mature peptide strings из train folds.

### 4.2 External peptide dictionaries

Добавление внешних peptide/AMP-баз сильно улучшило mature peptide detection.

Лучший Aho-only all-sources fixed rerun примерно:

| Setup | Pep F1 | Pep precision | Pep recall | All F1 | Propep F1 |
|---|---:|---:|---:|---:|---:|
| all sources | ~0.249 | ~0.674 | ~0.153 | ~0.127 | 0.000 |

Strict-режим:

| Setup | Pep F1 | Вывод |
|---|---:|---|
| all sources strict | ~0.097 | Всё ещё лучше UniProt-only; эффект не только exact test lookup |

Интерпретация:

- часть прироста - retrieval известных науке peptides;
- часть прироста остаётся после удаления exact eval gold sequences;
- внешние словари дают полезные overlapping / truncated / extended / near-boundary сигналы.

### 4.3 По источникам

| Источник | Наблюдение |
|---|---|
| DRAMP natural/general | Высокая точность, хороший clean peptide prior |
| dbAMP | Шире, больше recall, но шумнее |
| APD6 natural | Маленький, но точный источник |
| AMPDB | В текущем виде почти не дал пользы |
| UniProt PROPEP | Слабый propeptide signal, exact перенос почти отсутствует |

Главный вывод Aho-only:

```text
Aho-prior полезен как peptide-specific knowledge source,
но не решает PROPEP и не заменяет ESM2/CRF.
```

---

## 5. Aho features для нейросетевой модели

Реализован генератор:

```text
src/utils/embedding_generators/make_embeddings_aho.py
```

Он создаёт per-residue tensor `[L, D]`, где при текущем наборе источников:

```text
D = 76
```

Признаки включают:

- aggregate `pep` / `propep` features;
- `inside`, `start`, `end`;
- hit counts;
- max hit length;
- relative position inside hit;
- source count;
- multi-source indicator;
- start/end window and decay features;
- source-specific channels for DRAMP/dbAMP/APD/UniProt.

Сначала Aho embeddings сохранялись по UniProt AC, но затем naming был исправлен на:

```text
md5(sequence).pt
```

чтобы совпасть с ESM2 precomputed embeddings.

Итоговый merged input:

```text
[ESM2 | Aho] = [1280 + 76] = 1356 channels
```

---

## 6. Neural fusion эксперименты

### 6.1 Early fusion / TriBranchResidual

Схема:

```text
ESM2 1280 → projection → 256
Aho 76    → projection → 16 → up-project → 256
fused = ESM2_projected + gated small Aho residual
→ LSTMCNN → CRF
```

Результат:

| Model | Pep F1 | Propep F1 | All F1 | Вывод |
|---|---:|---:|---:|---|
| ESM2 baseline | ~0.602 | ~0.612 | ~0.608 | базовая точка |
| Early TriBranch | ~0.568 | ~0.616 | ~0.594 | хуже baseline |

Вывод: early fusion оказался неудачным. Вероятно, sparse Aho features слишком рано смешиваются с dense ESM2 representation, а ESM2 ещё и сжимается.

### 6.2 Late emission fusion

Схема:

```text
ESM2 → LSTMCNN → base emissions [B, L, 3]
Aho  → small Aho head → aho emissions [B, L, 3]
final emissions = base + scale * aho
→ CRF
```

Aho-head zero-initialized, поэтому стартует как обычный ESM2 baseline.

Результаты:

| Model | Pep F1 | Propep F1 | All F1 | Комментарий |
|---|---:|---:|---:|---|
| ESM2 baseline | ~0.602 | ~0.612 | ~0.608 | исходный baseline |
| Late fusion linear | ~0.594 | ~0.633 | ~0.615 | лучший all F1 из Aho-neural на тот момент |
| Late fusion h32 | ~0.595 | ~0.597 | ~0.596 | segment F1 хуже, но residue AUC лучше |

Вывод:

- late fusion намного лучше early fusion;
- linear-head дал лучший `all F1`, но улучшение пришло в основном через `PROPEP`, а `PEPTIDE` немного просел;
- h32 показал, что Aho может нести полезный residue-level signal, но segment decoding не улучшился.

### 6.3 Mid fusion

Схема:

```text
ESM2 → LSTMCNN → contextual hidden h_i
Aho → encoder / raw-normalized features a_i
[h_i ; a_i] → emission correction
→ CRF
```

Результаты:

| Model | Pep F1 | Propep F1 | All F1 | Комментарий |
|---|---:|---:|---:|---|
| Mid raw, all-labels | ~0.612 | ~0.599 | ~0.605 | впервые улучшил `PEPTIDE`, но просадил `PROPEP` |
| Mid raw, pep-only mask | ~0.584 | ~0.578 | ~0.581 | хуже baseline |

Вывод:

- mid fusion может лучше использовать Aho для mature peptide detection;
- простое masking “Aho влияет только на peptide logit” не сработало;
- даже peptide-only correction меняет CRF competition между `None`, `Peptide`, `Propeptide` и может косвенно вредить `PROPEP`.

### 6.4 True sparse transition bias

Была реализована sparse-версия input-dependent transition bias, чтобы не хранить огромный tensor `[B, L, 101, 101]`.

Идея:

```text
Aho start  → усилить переход None → PepStart
Aho inside → усилить internal peptide transitions
Aho end    → усилить PepEnd → None
```

Результат:

| Model | Pep F1 | Propep F1 | All F1 | Residue AUC all | Вывод |
|---|---:|---:|---:|---:|---|
| Sparse transition bias | ~0.543 | ~0.570 | ~0.558 | ~0.823 | Segment F1 сильно хуже, AUC лучше |

Learned biases оказались положительными почти для всех start/inside/end и даже сильнее для propep:

```text
pep_start    ~0.236
pep_inside   ~0.234
pep_end      ~0.308
propep_start ~0.342
propep_inside~0.272
propep_end   ~0.379
```

Вывод:

- настоящая transition-bias ветка слишком сильно меняет path dynamics CRF;
- она улучшает ранжирование residue probabilities, но портит Viterbi segment decoding;
- Aho лучше использовать как evidence/state prior, чем как direct controller of pairwise transitions.

### 6.5 Simple state-level boundary bias

После неудачи true transition bias был подготовлен более простой вариант:

```text
class emissions → repeat to CRF states
+ Aho state-level bias
→ обычный CRF
```

Он не меняет `multi_tag_crf.py`, а добавляет bias к CRF-state emissions:

- peptide start state;
- peptide internal states;
- peptide end state;
- optionally propep states;
- optionally weak pep-to-propep inside cross-bias.

Ближайшие эксперименты:

1. `PEPTIDE boundary-only`: только start/end peptide state bias.
2. `pep → propep weak inside`: mature peptide hit как слабое evidence того, что он может лежать внутри более широкого propeptide region.

---

## 7. Обсуждение PROPEP

Была уточнена мысль: утверждение “Aho бесполезен для propep” слишком грубое.

Более точная формулировка:

```text
External mature peptide dictionaries не дают прямого PROPEP signal,
но known peptide hit может быть косвенным evidence для PROPEP context.
```

Причина: mature peptide может совпадать с целевым `PEPTIDE`, но также может быть подотрезком более широкого processing/propeptide region.

Поэтому предложен weak cross-bias:

```text
pep Aho inside feature → weak propep inside state bias
```

Но не стоит жёстко переносить peptide boundaries на propeptide boundaries, потому что границы mature peptide и propeptide обычно не обязаны совпадать.

Возможные источники/правила для PROPEP:

- UniProt `FT PROPEP` - основной прямой источник;
- более широкий UniProt-derived set по reviewed entries with PROPEP;
- MEROPS/TopFIND - скорее cleavage/bond, не direct PROPEP labels;
- ProP-like / furin / proprotein convertase predictors - useful features, not universal labels;
- basic residue motifs: `KR`, `RR`, `RK`, `KK`, furin-like `R-X-[K/R]-R`;
- signal peptide / secretory context;
- distance to known mature peptide hit.

---

## 8. Новая идея: bond signal + distance-aware bond loss

Была предложена новая линия: использовать внешний bond/cleavage signal от отдельной модели как дополнительную информацию, а дополнительно учить auxiliary loss для границ собственных UniProt PEPTIDE/PROPEP.

Ключевая мысль: external cleavage model не должна становиться hard target для нашей задачи. Лучше использовать её как feature / prior.

### 8.1 External bond signal как feature

Внешняя cleavage/bond model даёт:

```text
cleavage_score[i] = вероятность разреза между residue i и i+1
```

Длина bond vector:

```text
L - 1
```

Для подачи в residue-level модель можно превратить это в признаки длины `L`:

```text
cleavage_before[pos]
cleavage_after[pos]
max_cleavage_score_window3
max_cleavage_score_window5
distance_to_nearest_high_score_cleavage
```

И дальше использовать как отдельную ветку:

```text
[ESM2 | Aho | BondFeatures]
```

Сначала лучше проверить:

```text
ESM2 + BondFeatures
```

и только потом:

```text
ESM2 + Aho + BondFeatures
```

### 8.2 Auxiliary bond head на UniProt boundaries

Для каждого annotated segment `[start, end]` строим boundary targets:

```text
left boundary:  between start-1 and start
right boundary: between end and end+1
```

В 0-based bond index:

```text
left_bond  = start - 2, если start > 1
right_bond = end - 1, если end < L
```

Модель:

```text
ESM2/LSTMCNN hidden h_i
bond_input[i] = concat(h_i, h_{i+1}, |h_{i+1}-h_i|)
bond_logit[i] = MLP(bond_input[i])
```

Loss:

```text
total_loss = CRF_loss + lambda_bond * bond_loss
```

### 8.3 Distance-aware smooth target

Вместо hard target `1 только ровно на границе` предлагается гладкий target через расстояние до ближайшей истинной границы:

```text
d(i) = min_b |i - b|
y_soft[i] = exp(-d(i) / tau)
```

или Gaussian:

```text
y_soft[i] = exp(-d(i)^2 / (2 sigma^2))
```

С окном:

```text
if d(i) > W: y_soft[i] = 0
```

Стартовые параметры:

```text
W = 5
tau = 1.5
lambda_bond = 0.02 / 0.05 / 0.1
positive weight alpha = 10-20
```

Weighted BCE:

```text
bond_loss = BCEWithLogitsLoss(bond_logits, y_soft, weight=1 + alpha * y_soft)
```

Смысл:

```text
prediction at exact boundary → best
prediction at ±1 residue → small penalty
prediction far away → strong penalty / background
```

### 8.4 Alternative: local distribution loss

Для каждой истинной границы `b` взять окно `[b-W, ..., b+W]` и задать distribution:

```text
q(i) ∝ exp(-|i-b| / tau)
p(i) = softmax(bond_logits_window)
loss = cross_entropy(q, p)
```

Это ещё ближе к идее “попал рядом - почти правильно”, но сложнее реализуется, особенно при пересекающихся окнах.

### 8.5 Рекомендуемый roadmap для bond loss

1. Реализовать target-only auxiliary binary soft bond loss на UniProt boundaries.
2. Проверить `ESM2 + bond loss` без Aho.
3. Проверить `ESM2 + Aho + bond loss`.
4. Добавить external cleavage/bond predictor scores как input features.
5. Только потом рассматривать teacher loss или multi-task batches с MEROPS/TopFIND.

---

## 9. Текущий статус и ближайшие эксперименты

Уже подготовлено/реализовано:

- Aho-only pipeline;
- external peptide dictionaries preprocessing;
- Aho feature generator;
- merge ESM2 + Aho embeddings;
- early fusion;
- late emission fusion;
- mid fusion;
- label-scale masking;
- true sparse transition bias;
- simple state-level boundary bias;
- proposed design for bond loss.

Лучший завершённый Aho-neural результат по `all F1` пока:

```text
late emission linear: all F1 ≈ 0.615
```

Лучший peptide-specific neural Aho result:

```text
mid raw all-labels: pep F1 ≈ 0.612
```

True transition bias пока отрицательный по segment F1, несмотря на высокий residue AUC.

Ближайшие запуски:

1. `lstmcnncrf_aho_state_bias` peptide boundary-only.
2. `lstmcnncrf_aho_state_bias` weak pep-to-propep inside cross-bias.
3. Затем - bond loss prototype на UniProt boundaries.

---

## 10. Главный вывод

Aho-prior оказался содержательным, но его нельзя использовать как standalone segmentation model.

Рабочая формулировка:

```text
Aho даёт peptide-specific interval evidence.
Его лучше использовать как мягкий prior/features для ESM2/CRF,
а не как замену модели и не как жёсткий контроллер CRF transitions.
```

Наиболее перспективные направления дальше:

```text
1. Simple state-level Aho boundary bias.
2. Aho + residue/bond features как отдельные late/mid fusion ветки.
3. Auxiliary distance-aware bond loss по UniProt boundaries.
4. External cleavage predictor scores как weak input features, не как hard labels.
```
