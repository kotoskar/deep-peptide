# Краткий отчёт по Aho-prior экспериментам

## 1. Исходные идеи

В начале обсуждения были предложены две основные идеи.

### Идея 1: Aho–Corasick prior

Прогонять белок через Aho–Corasick со словарём известных биологически активных пептидов/пропептидов и передавать модели информацию о том, какие подотрезки уже известны как активные.

Ожидание:

- модель лучше найдёт границы зрелых пептидов;
- модель сможет выбирать из уже известных биологически активных вариантов;
- сигнал может помочь там, где ESM2/CRF неуверенны.

Риск:

- модель может стать менее “креативной”;
- часть прироста может быть retrieval известных test-пептидов;
- внешние базы в основном содержат mature peptides, а не propeptides.

### Идея 2: bond / cleavage loss

Добавить auxiliary loss не по сегментации, а по каждой паре соседних остатков:

```text
между residue_i и residue_{i+1} есть разрез или нет
```

Для этого потенциально подходят внешние cleavage-site базы вроде MEROPS, TopFIND, DICED, CutDB/PMAP, Proteasix.

Эта идея пока не реализовывалась, но остаётся перспективной для следующего этапа.

---

## 2. Aho-only pipeline

Был реализован отдельный Aho-only baseline.

### Preprocessing

Были приведены к единому формату источники:

```text
data/aho_train/uniprot_2022.tsv
data/aho_train/dramp_general.tsv
data/aho_train/dramp_natural.tsv
data/aho_train/dbamp_3.tsv
data/aho_train/apd6_natural.tsv
data/aho_train/ampdb.tsv
```

Формат источников:

```text
sequence    label    ...
```

где:

```text
label ∈ {pep, propep}
```

`source` берётся из имени файла.

### train_loop_aho.py

Был реализован `train_loop_aho.py`, который:

- читает белки из `labeled_sequences.csv`;
- читает fold split из `graphpart_assignments.csv`;
- загружает Aho-словари из `data/aho_train/`;
- для folded-источников использует только нужные folds;
- для внешних источников без fold использует весь файл;
- дедуплицирует последовательности;
- на validation подбирает параметры scoring/overlap resolution;
- на test делает prediction;
- считает segment-level metrics.

### Важный bugfix

В ранней версии matching не проверял `protein_id`, то есть prediction на одном белке мог засчитаться как TP для gold-сегмента на другом белке, если label и координаты были близкими.

После исправления matching стал protein-aware:

```python
if gseg.protein_id != pseg.protein_id:
    continue
```

Все выводы ниже основаны на fixed/rerun метриках.

---

## 3. Aho-only результаты

### UniProt-only

Aho-only на train-fold UniProt сегментах почти не работает на GraphPart test split.

```text
pep F1 ≈ 0.029
all F1 ≈ 0.013
```

Вывод:

> Простое запоминание train PEPTIDE/PROPEP сегментов почти не переносится на test fold.

Это хороший отрицательный baseline: DeepPeptide/ESM2 не может быть объяснён простым memorization train-пептидов.

---

### Внешние peptide dictionaries

После добавления внешних источников Aho-only стал заметно лучше для mature peptides.

Лучший all-sources Aho-only результат был примерно:

```text
pep F1        ≈ 0.249
pep precision ≈ 0.674
pep recall    ≈ 0.153
```

Strict-режим, где exact eval gold peptides удалялись из внешних источников, всё ещё был лучше UniProt-only:

```text
strict pep F1 ≈ 0.097
```

Вывод:

> Существенная часть normal-прироста — retrieval известных пептидов, но strict-результат показывает, что внешние базы дают и полезный near/overlap prior.

---

### Propeptide

Для `propep` Aho-only практически не помогает.

```text
propep F1 ≈ 0
```

Вывод:

> Внешние AMP/peptide базы дают сигнал почти исключительно для `pep`, но не для `propep`.

---

## 4. Источники данных: что оказалось полезным

### DRAMP natural/general

DRAMP оказался одним из самых чистых источников.

Характеристика:

- высокая precision;
- хороший peptide prior;
- лучше использовать как сильный источник для `pep`.

### dbAMP

dbAMP даёт больше coverage, но более шумный.

Характеристика:

- выше recall;
- ниже precision;
- полезен как soft feature;
- dbAMP-only hit стоит считать слабее, чем multi-source hit.

### APD6 natural

APD6 меньше по объёму, но очень чистый.

Характеристика:

- high precision;
- хороший source-specific signal;
- особенно полезен, если hit подтверждён вместе с DRAMP/dbAMP.

### AMPDB

AMPDB в текущем виде почти не дал пользы.

Возможные причины:

- слабое пересечение с задачей;
- шумный состав;
- preprocessing мог оставить мало релевантных canonical 5–50 sequences.

### Практический вывод по источникам

```text
DRAMP/APD/dbAMP overlap = сильный Aho-сигнал
dbAMP-only              = более шумный сигнал
UniProt propep          = слабый отдельный сигнал
AMPDB                   = пока не главный источник
```

---

## 5. Генерация Aho features

Был реализован генератор:

```text
src/utils/embedding_generators/make_embeddings_aho.py
```

Он генерирует per-residue Aho features для каждого белка.

### Признаки

На каждую позицию строятся признаки:

```text
inside
start
end
count_log
max_len_norm
rel_from_start
rel_to_end
source_count_log
multi_source
start_window3
end_window3
start_decay
end_decay
```

Также добавлены source-specific признаки для каждого source/label.

Итоговая размерность в текущем наборе:

```text
AHO_DIM = 76
```

### Naming fix

Сначала Aho embeddings сохранялись по `AC/protein_id`, но ESM2 embeddings были сохранены по `md5(sequence)`.

Генератор был исправлен: теперь Aho `.pt` сохраняются по `md5(sequence).pt`, чтобы совпадать с ESM2 embeddings.

### Merge

После генерации Aho features они склеиваются с ESM2:

```text
[ESM2 | Aho] = 1280 + 76 = 1356
```

---

## 6. Neural experiments: Early fusion / TriBranch

Первый способ интеграции Aho был early fusion через `TriBranchResidual`.

Схема:

```text
ESM2 1280 → projector → 256
Aho 76    → projector → 16 → 256
fused = ESM2_projected + gated_residual(Aho)
```

Результат:

```text
ESM2 baseline all F1      ≈ 0.608
TriBranch ESM2+Aho all F1 ≈ 0.594
```

Также упал `pep F1`.

Вывод:

> Early fusion оказался неудачным. Вероятно, проблема не в Aho-сигнале, а в том, что ESM2-представление сжималось/искажалось, а sparse Aho features слишком рано смешивались с dense embeddings.

---

## 7. Neural experiments: Late emission fusion

Была добавлена модель:

```text
lstmcnncrf_aho_emission_fusion
```

Схема:

```text
ESM2 → LSTMCNN → base emissions
Aho  → small Aho head → aho emissions
final emissions = base emissions + aho_scale * aho emissions
final emissions → CRF
```

Aho-head был zero-initialized, то есть модель стартовала как обычная ESM2-модель.

### Linear head

Результат:

```text
pep F1     ≈ 0.594
propep F1  ≈ 0.633
all F1     ≈ 0.615
```

Против ESM2 baseline:

```text
pep F1     ≈ 0.602
propep F1  ≈ 0.612
all F1     ≈ 0.608
```

Вывод:

> Late emission fusion улучшил общий F1, но в основном за счёт роста propep precision/F1. При этом peptide recall снизился, и `pep F1` стал ниже baseline.

### Hidden size 32

Результат:

```text
all F1 ≈ 0.596
```

Но residue-level AUC стал лучше.

Вывод:

> Нелинейная Aho-head умеет извлекать полезный residue-level signal, но class-level emission fusion не переводит его в лучший segment-level F1.

---

## 8. Neural experiments: Mid fusion

Была добавлена модель:

```text
lstmcnncrf_aho_mid_fusion
```

Схема:

```text
ESM2 → LSTMCNN → contextual hidden features
Aho  → encoder / raw normalized Aho features
[hidden ; Aho] → emission correction
final emissions → CRF
```

Это промежуточный вариант между early fusion и late emission fusion:

- Aho не портит вход ESM2;
- но Aho используется вместе с contextual hidden state перед final emissions.

### Mid fusion без сжатия Aho

Результат:

```text
pep F1     ≈ 0.612
propep F1  ≈ 0.599
all F1     ≈ 0.605
```

Вывод:

> Mid fusion впервые улучшил именно `pep F1`, что согласуется с Aho-only результатами. Но одновременно ухудшился `propep`, поэтому общий F1 остался чуть ниже ESM2 baseline.

### Label masking

Для mid/late fusion был добавлен masking Aho-поправки:

```text
--aho_none_scale
--aho_pep_scale
--aho_propep_scale
```

Например, peptide-only correction:

```bash
--aho_none_scale 0.0 \
--aho_pep_scale 1.0 \
--aho_propep_scale 0.0
```

Гипотеза:

> Aho помогает `pep`, но мешает `propep`, потому что внешние словари в основном peptide-only.

Соответственно, `pep-only` masking должен сохранить прирост по peptide и уменьшить просадку propeptide.

---

## 9. CRF transition bias

Поскольку Aho-hit — это интервальный сигнал, а не просто residue-level feature, был подготовлен следующий режим:

```text
lstmcnncrf_aho_transition_bias
```

Суть:

```text
Aho-hit [s, e] добавляет input-dependent bias к CRF transitions:

None → PepStart
PepInternal → PepInternal
PepEnd → None
```

Это уже ближе к исходной идее: модель получает не просто информацию “позиция похожа на peptide”, а информацию:

```text
здесь вероятный старт
здесь вероятный внутренний участок
здесь вероятный конец
```

Для этого был подготовлен modified `multi_tag_crf.py`, где CRF может принимать `transition_bias[B, L, S, S]`.

Результаты этого эксперимента пока не получены.

---

## 10. Общие выводы

### Aho-only

Aho-only не является конкурентом ESM2/CRF, но показывает, что known-peptide prior реален и полезен для mature peptide detection.

### Внешние источники

DRAMP, dbAMP и APD дают полезный сигнал для `pep`, но почти не помогают `propep`.

### Early fusion

Неудачна. Скорее всего, из-за искажения ESM2 representation и слишком раннего смешивания sparse Aho features с dense embeddings.

### Late fusion

Лучше early fusion. Дал лучший `all F1` среди завершённых Aho-neural вариантов, но улучшение пришло в основном через `propep`, а не через `pep`.

### Mid fusion

Лучше использует Aho для `pep`, но просаживает `propep`. Нужен `pep-only` masking.

### Transition bias

Самый биологически осмысленный следующий тест, потому что Aho — это интервальный/границевый prior.

---

## 11. Практический статус

Уже реализовано:

```text
Aho-only baseline
preprocessing external peptide dictionaries
Aho feature generator
ESM2+Aho embedding merge
late emission fusion
mid fusion
label-scale masking
transition-bias CRF branch
```

Ключевые текущие результаты:

```text
ESM2 baseline all F1        ≈ 0.608
TriBranch early fusion      ≈ 0.594
Late emission fusion linear ≈ 0.615
Mid fusion raw Aho          ≈ 0.605
Aho-only all-sources pep F1 ≈ 0.249
```

Главный научный вывод на текущий момент:

> Aho-prior полезен, но его нельзя просто склеить с ESM2 embeddings. Он работает как peptide-specific structured prior. Наиболее перспективные направления — `pep-only mid fusion` и CRF transition/boundary bias.
