# Резюме диалога: Aho / state bias / bond loss / boundary-state experiments

Дата: 2026-05-11

## 1. Контекст задачи

Работа ведётся над предсказанием сегментации precursor-белка на:

- `PEPTIDE` — зрелые / биологически активные пептиды;
- `PROPEP` — пропептидные участки;
- фон / остальная последовательность.

Базовая схема модели:

```text
ESM2 residue embeddings [L, 1280]
→ LSTMCNN
→ coarse emissions: None / PEPTIDE / PROPEP
→ repeat emissions into multistate CRF states
→ constrained CRF loss / Viterbi decoding
→ predicted segments
```

Важная особенность baseline: нейросеть выдаёт только coarse logits, а различие `start / inside / end` кодируется CRF state-space.

Ориентировочный baseline из предыдущих экспериментов:

| Model | Pep F1 | Propep F1 | All F1 |
|---|---:|---:|---:|
| ESM2 baseline | ~0.602 | ~0.612 | ~0.608 |

Лучшие ранние Aho-neural результаты:

| Model | Pep F1 | Propep F1 | All F1 |
|---|---:|---:|---:|
| Late emission linear | ~0.594 | ~0.633 | ~0.615 |
| Mid raw all-labels | ~0.612 | ~0.599 | ~0.605 |
| Sparse transition bias | ~0.543 | ~0.570 | ~0.558 |

Основной вывод до новых запусков: Aho полезен как мягкий evidence/prior, но плохо работает как жёсткий контроллер CRF transitions.

---

## 2. State bias experiment

Был рассмотрен эксперимент с `state bias`, где Aho-сигнал добавлялся к CRF-state emissions, а не к pairwise transitions.

Результат:

| Metric | PEPTIDE | PROPEP | ALL |
|---|---:|---:|---:|
| Precision | 0.660 | 0.680 | 0.671 |
| Recall | 0.520 | 0.537 | 0.530 |
| F1 | 0.582 | 0.600 | 0.592 |
| Residue MCC | 0.664 | 0.736 | 0.734 |
| Residue ROC-AUC | 0.852 | 0.595 | 0.760 |

Вывод:

- `state bias` оказался лучше, чем true sparse transition bias по segment F1;
- но всё ещё хуже baseline и хуже late emission fusion;
- модель стала консервативной: precision неплохой, recall низкий;
- residue-level signal есть, но он плохо превращается в корректные Viterbi-сегменты.

Интерпретация:

```text
Aho/state bias даёт полезный ranking/evidence signal,
но прямое вмешательство в CRF-state path всё ещё портит segment decoding.
```

---

## 3. Soft bond loss: что было реализовано

Была добавлена новая ветка:

```bash
--model lstmcnncrf_bond_loss
```

Изменённые файлы:

- `src/train_loop_crf.py`
- `src/models/__init__.py`
- `src/models/crf_models.py`

Идея:

```text
LSTMCNN hidden features h_i
   ├─ emission head → CRF loss по gold multistate labels
   └─ bond head     → soft BCE loss по gold boundary targets
```

Итоговый loss:

```text
total_loss = CRF_loss + lambda_bond * bond_loss
```

### Что такое bond

`bond_i` — это связь между соседними остатками:

```text
bond i = boundary between residue i and residue i+1
```

Bond-head получает:

```text
[h_i, h_{i+1}, |h_i - h_{i+1}|]
```

и предсказывает logit для наличия границы между этими двумя residues.

### Как строится gold bond target

Gold boundary извлекается из multistate CRF labels. Граница ставится при переходах:

```text
None -> PEPTIDE
PEPTIDE -> None
None -> PROPEP
PROPEP -> None
PEPTIDE -> PROPEP
PROPEP -> PEPTIDE
```

Также учитываются соседние сегменты одного типа без фона:

```text
PEPTIDE_end -> PEPTIDE_start
PROPEP_end -> PROPEP_start
```

Если gold segment задан в 1-based координатах `[start, end]`, то:

```text
left boundary  = bond index start - 2, если start > 1
right boundary = bond index end - 1,   если end < L
```

Расстояние считается между bond-индексами:

```text
d(candidate_bond, true_boundary_bond) = |j - b|
```

Soft target:

```text
y[j] = max_b exp(-|j - b| / tau)
```

или Gaussian:

```text
y[j] = max_b exp(-(j - b)^2 / (2 * tau^2))
```

С окном `bond_soft_window`; вне окна target равен 0.

---

## 4. Auxiliary-only bond loss result

Запуск:

```bash
python run.py \
  --model lstmcnncrf_bond_loss \
  --embedding precomputed \
  --embeddings_dir data/uniprot_2022/embeddings/embeddings_esm2 \
  --embedding_dim 1280 \
  --seq_input_size 1280 \
  --data_file data/uniprot_2022/labeled_sequences.csv \
  --partitioning_file data/uniprot_2022/graphpart_assignments.csv \
  --label_type multistate_with_propeptides \
  --feature_extractor LSTMCNN \
  --out_dir runs/esm2_bond_loss_soft_l005_w5_tau15 \
  --checkpoints_dir runs/esm2_bond_loss_soft_l005_w5_tau15/checkpoints \
  --bond_loss_lambda 0.05 \
  --bond_soft_window 5 \
  --bond_soft_tau 1.5 \
  --bond_soft_mode exp \
  --bond_positive_weight 10 \
  --bond_hidden_size 64
```

Результат:

| Metric | PEPTIDE | PROPEP | ALL |
|---|---:|---:|---:|
| Precision | 0.603 | 0.651 | 0.629 |
| Recall | 0.494 | 0.511 | 0.503 |
| F1 | 0.543 | 0.573 | 0.559 |
| Residue MCC | 0.664 | 0.689 | 0.720 |
| Residue ROC-AUC | 0.860 | 0.632 | 0.785 |

Вывод:

- segment F1 сильно хуже baseline;
- residue ROC-AUC неплохой, особенно для PEPTIDE;
- bond loss действительно учит boundary/residue signal;
- но этот signal не участвует в CRF decoding, потому что bond-head используется только как auxiliary loss.

Главный вывод:

```text
Auxiliary-only bond loss недостаточен:
модель учит границы, но Viterbi decoder напрямую их не видит.
```

---

## 5. Boundary-state + bond loss: новая модификация

После неудачи auxiliary-only bond loss была сделана более сильная архитектурная модификация:

```bash
--model lstmcnncrf_boundary_bond_loss
```

### Что такое boundary

`boundary` здесь — это learned state-level signal, предсказываемый из LSTMCNN hidden features:

```text
h_i → boundary_state_head(h_i)
    → pep_start, pep_inside, pep_end,
      propep_start, propep_inside, propep_end
```

Он добавляется прямо к CRF-state emissions:

```text
pep_start  → state 1
pep_inside → states 1..50
pep_end    → state 50

propep_start  → state 51
propep_inside → states 51..100
propep_end    → state 100
```

Отличие от baseline:

```text
baseline:
PEPTIDE logit просто копируется во все peptide states 1..50

boundary version:
PEPTIDE logit копируется во все peptide states,
но сверху добавляются разные learned поправки для start / inside / end states
```

### Что значит название boundary_bond

```text
boundary = learned state-level start/inside/end bias для CRF decoding
bond     = auxiliary soft BCE loss по межостаточным границам
```

То есть:

```text
ESM2
→ LSTMCNN
→ hidden features h_i
   ├─ обычная emission head: None / PEPTIDE / PROPEP
   ├─ boundary head: start / inside / end state corrections
   └─ bond head: auxiliary bond loss
→ CRF
```

---

## 6. Boundary + bond result

Запуск с `boundary_state_scale=1.0` и `bond_loss_lambda=0.02`:

```bash
python run.py \
  --model lstmcnncrf_boundary_bond_loss \
  --embedding precomputed \
  --embeddings_dir data/uniprot_2022/embeddings/embeddings_esm2 \
  --embedding_dim 1280 \
  --seq_input_size 1280 \
  --data_file data/uniprot_2022/labeled_sequences.csv \
  --partitioning_file data/uniprot_2022/graphpart_assignments.csv \
  --label_type multistate_with_propeptides \
  --feature_extractor LSTMCNN \
  --out_dir runs/esm2_boundary_bond_l002_w5_tau15 \
  --checkpoints_dir runs/esm2_boundary_bond_l002_w5_tau15/checkpoints \
  --boundary_state_hidden_size 64 \
  --boundary_state_dropout 0.1 \
  --boundary_state_scale 1.0 \
  --bond_loss_lambda 0.02 \
  --bond_soft_window 5 \
  --bond_soft_tau 1.5 \
  --bond_soft_mode exp \
  --bond_positive_weight 10 \
  --bond_hidden_size 64 \
  --bond_dropout 0.1
```

Результат:

| Metric | PEPTIDE | PROPEP | ALL |
|---|---:|---:|---:|
| Precision | 0.596 | 0.781 | 0.681 |
| Recall | 0.567 | 0.529 | 0.546 |
| F1 | 0.581 | 0.631 | 0.606 |
| Residue MCC | 0.702 | 0.674 | 0.739 |
| Residue ROC-AUC | 0.794 | 0.520 | 0.689 |

Вывод:

- модель почти вернулась к baseline по `all F1`;
- `PROPEP F1` стал выше baseline и выше ранних boundary/bond ablations;
- `PEPTIDE F1` просел;
- propeptide precision стал очень высоким, но recall низкий.

Интерпретация:

```text
Boundary + bond помогает PROPEP,
но делает модель слишком консервативной и ухудшает PEPTIDE recall/quality.
```

---

## 7. Boundary-only result

Был запущен ablation без bond loss, то есть `bond_loss_lambda=0.0`.

Результат:

| Metric | PEPTIDE | PROPEP | ALL |
|---|---:|---:|---:|
| Precision | 0.649 | 0.746 | 0.697 |
| Recall | 0.542 | 0.506 | 0.522 |
| F1 | 0.591 | 0.603 | 0.597 |
| Residue MCC | 0.663 | 0.697 | 0.738 |
| Residue ROC-AUC | 0.819 | 0.513 | 0.697 |

Вывод:

- без bond loss PEPTIDE чуть лучше, чем в boundary+bond;
- но PROPEP заметно хуже;
- общий `all F1` ниже, чем у boundary+bond;
- значит bond loss всё-таки полезен, но в текущей форме в основном для PROPEP.

Сравнение:

| Model | Pep F1 | Propep F1 | All F1 |
|---|---:|---:|---:|
| Boundary-only | 0.591 | 0.603 | 0.597 |
| Boundary + bond | 0.581 | 0.631 | 0.606 |

---

## 8. Общая картина по новым экспериментам

| Experiment | Pep F1 | Propep F1 | All F1 | Main observation |
|---|---:|---:|---:|---|
| ESM2 baseline | ~0.602 | ~0.612 | ~0.608 | базовая точка |
| Aho late emission linear | ~0.594 | ~0.633 | ~0.615 | лучший ранний all F1 |
| Aho mid raw all-labels | ~0.612 | ~0.599 | ~0.605 | лучший PEPTIDE среди Aho-neural |
| Sparse transition bias | ~0.543 | ~0.570 | ~0.558 | transition control портит decoding |
| Simple state bias | 0.582 | 0.600 | 0.592 | high precision / low recall |
| Auxiliary bond loss only | 0.543 | 0.573 | 0.559 | хороший residue signal, плохой segment decoding |
| Boundary-only | 0.591 | 0.603 | 0.597 | boundary помогает, но не пробивает baseline |
| Boundary + bond | 0.581 | 0.631 | 0.606 | сильный PROPEP, слабее PEPTIDE |

Главный вывод:

```text
Новые идеи не бесполезны: они дают понятные сигналы.
Но каждая в одиночку двигает разные части задачи:

Aho/mid-fusion лучше помогает PEPTIDE.
Boundary/bond лучше помогает PROPEP.
Auxiliary-only bond улучшает residue ranking, но не segment decoding.
State/transition вмешательство легко делает модель слишком консервативной.
```

---

## 9. Текущая проблема

Почти все новые ветки страдают от одного паттерна:

```text
precision растёт или остаётся высоким,
recall проседает,
segment F1 не растёт.
```

Особенно заметно в boundary/bond:

```text
propep precision = 0.781
propep recall    = 0.529
```

Это значит, что модель научилась выбирать более уверенные сегменты, но пропускает слишком много истинных сегментов.

---

## 10. Практический следующий шаг

Сначала стоит проверить ослабленный boundary/bond:

```bash
python run.py \
  --model lstmcnncrf_boundary_bond_loss \
  --embedding precomputed \
  --embeddings_dir data/uniprot_2022/embeddings/embeddings_esm2 \
  --embedding_dim 1280 \
  --seq_input_size 1280 \
  --data_file data/uniprot_2022/labeled_sequences.csv \
  --partitioning_file data/uniprot_2022/graphpart_assignments.csv \
  --label_type multistate_with_propeptides \
  --feature_extractor LSTMCNN \
  --out_dir runs/esm2_boundary_bond_l001_scale05 \
  --checkpoints_dir runs/esm2_boundary_bond_l001_scale05/checkpoints \
  --boundary_state_hidden_size 64 \
  --boundary_state_dropout 0.1 \
  --boundary_state_scale 0.5 \
  --bond_loss_lambda 0.01 \
  --bond_soft_window 5 \
  --bond_soft_tau 1.5 \
  --bond_soft_mode exp \
  --bond_positive_weight 10 \
  --bond_hidden_size 64 \
  --bond_dropout 0.1
```

Цель: снизить чрезмерную консервативность decoding и вернуть recall.

---

## 11. Наиболее перспективная следующая модификация

Если ослабленный boundary/bond не даст прироста хотя бы до `0.615–0.62 all F1`, наиболее логичный следующий шаг:

```text
AhoMidFusion + BoundaryState + слабый или нулевой BondLoss
```

Архитектура:

```text
ESM2 → LSTMCNN → hidden features h_i

Aho branch:
Aho features + h_i → emission correction

Boundary branch:
h_i → start / inside / end state correction

Optional bond branch:
[h_i, h_{i+1}, |h_i-h_{i+1}|] → auxiliary bond loss

final emissions → CRF
```

Причина:

```text
Aho ранее давал лучший PEPTIDE signal.
Boundary/bond сейчас дал лучший PROPEP signal.
Их сочетание может быть сильнее, чем каждая ветка отдельно.
```

---

## 12. Краткий итог

1. `state bias` подтвердил, что прямое biasing CRF-state path даёт сигнал, но делает модель консервативной.
2. `auxiliary-only bond loss` улучшает residue/boundary signal, но не помогает segment F1, потому что bond signal не участвует в decoding.
3. `boundary-state head` — правильнее, потому что даёт CRF явные learned поправки для `start / inside / end`.
4. `boundary + bond` почти догнал baseline и улучшил `PROPEP`, но ухудшил `PEPTIDE`.
5. Следующий сильный кандидат — гибрид `AhoMidFusion + BoundaryState`, возможно со слабым bond loss.

Рабочая формулировка для отчёта:

```text
Boundary-aware state emissions partially address the limitation of the original DeepPeptide-style CRF, where the same coarse peptide logit is repeated across all start/internal/end states. The learned boundary head improves propeptide segment precision and F1, especially when regularized with a soft auxiliary bond loss. However, the current variant remains conservative and does not yet improve the overall F1 over the strongest Aho late-fusion baseline. This suggests that boundary-aware decoding is useful, but should be combined with Aho-based peptide evidence to recover mature peptide recall and improve total segment-level F1.
```
