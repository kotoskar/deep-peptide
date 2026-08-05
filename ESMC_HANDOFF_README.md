# Запуск esmc6b nested-CV на внешнем сервере (A100/V100)

Что нужно посчитать: 3 модели × 20 ячеек (5 outer × 4 inner) = 60 обучений.
Модели: `esmc6b_boundary`, `esmc6b_adapter_only`, `esmc6b_full`.
(`esmc6b_plain` считается отдельно локально, сюда не входит.)

Каждая ячейка — реальное обучение на 100 эпох, ~10-11 часов на одну ячейку
(измерено локально и на A100 — эта модель маленькая и упирается не в компут,
а в память/bandwidth, так что скорость не сильно зависит от конкретной карты).

## Шаг 0: распаковать

```bash
tar -xf esmc6b_handoff.tar
cd esmc6b_handoff
```

Проверить, что распаковалось разумно:
```bash
du -sh data/uniprot_2022/embeddings/embeddings_esmc6b   # должно быть ~20GB, 9086 файлов
ls runs/2026_esmc6b_boundary/config.json runs/2026_esmc6b_adapter_only/config.json runs/2026_esmc6b_full/config.json
```

## Шаг 1: окружение

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Проверка, что PyTorch реально видит GPU (важно — иногда `pip install torch==X`
ставит билд под другую версию CUDA, чем на сервере):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Если `torch.cuda.is_available()` выдаёт `False` — GPU не подхвачен, дальше
двигаться нет смысла, нужно поставить PyTorch под версию CUDA сервера
(смотреть `nvidia-smi` -> driver/CUDA version -> https://pytorch.org, взять
оттуда правильную команду `pip install torch==...`).

## Шаг 2: быстрая проверка (~5-10 минут)

Прогнать одну ячейку на 1 эпоху, чтобы поймать проблемы окружения (не хватает
пакета, не тот путь и т.д.) прежде чем тратить время на что-то большее:

```bash
PYTHONPATH=. python3 analysis/experiments/train_nested_cell.py \
  --base runs/2026_esmc6b_boundary/config.json \
  --emb data/uniprot_2022/embeddings/embeddings_esmc6b \
  --split data/uniprot_2026/graphpart_assignments_5motif.esmc6bcovered.csv \
  --out sanity_check --outer 0 --inner 1 --n_folds 5 --set epochs=1
```

Должно отработать без ошибок и написать `runs/sanity_check/outer0_inner1/cell_result.json`.
Если всё ок — можно удалить: `rm -rf runs/sanity_check`.

Если тут что-то падает — присылайте лог, не идите дальше, дешевле поймать
проблему на 1 эпохе, чем на 4-часовом прогоне.

## Шаг 3 (опционально): автоподбор параллельности (~15-30 минут)

В архиве уже лежит `probe_result.json` с безопасным дефолтом `recommended_concurrency: 1`
(проверено — эта же модель спокойно тянет 1x локально на 16GB карте). Если
спешите или не хочется возиться — можно сразу перейти к Шагу 4, будет просто
медленнее (без параллелизма).

Если время есть — стоит один раз замерить реальный потолок для конкретной
GPU (A100 или V100, память неизвестна) и получить ускорение в 2-6 раз.
Скрипт сам поднимает 1, 2, 3, 4, 6, 8, 12, 16, 20 одновременных ячеек
(короткими 4-минутными прогонами, не полное обучение), останавливается на
первом OOM и перезаписывает `probe_result.json` рекомендованным значением
(на 1 ступень ниже потолка — с запасом).

```bash
bash analysis/experiments/probe_concurrency.sh
```

В конце в терминале будет что-то вроде:
```
Highest concurrency with NO failures: 4x
Recommended (with safety margin):     3x
Written to probe_result.json
```

Можно открыть `probe_result.json` и поменять `recommended_concurrency` руками,
если хочется быть ещё осторожнее (например, если на сервере не только ваша
задача и не хочется зря отжирать всю память).

## Шаг 4: основной прогон (несколько дней)

```bash
nohup bash analysis/experiments/run_esmc_queue.sh > run_esmc_queue.log 2>&1 &
disown
```

Это займёт `60 / recommended_concurrency` партий, каждая партия — реальное
обучение до 100 эпох (~10-11 часов). Например, при recommended=3: 20 партий ×
~11ч ≈ 9 дней. При recommended=6: ~10 партий ≈ 4.5 дня.

Прогресс смотреть так:
```bash
tail -f run_esmc_queue.log
cat logs/timing.txt                        # тайминги партий
find runs/5cv_esmc6b_* -name cell_result.json | wc -l   # сколько ячеек готово из 60
```

## Если сервер перезагрузили / процесс оборвался

Ничего страшного, ничего не портится:
- уже готовые ячейки (`cell_result.json` есть) скрипт пропустит сам;
- недоделанная ячейка возобновится с последнего чекпоинта (сохраняется каждые
  10 эпох) — теряется максимум ~9 эпох одной ячейки, не весь прогон.

Просто перезапустите ту же команду:
```bash
nohup bash analysis/experiments/run_esmc_queue.sh > run_esmc_queue.log 2>&1 &
disown
```

## Что забрать обратно

Когда всё (или частично) готово — обратно нужны только эти папки (эмбеддинги
30GB+ отправлять обратно не нужно, это входные данные, не результат):

```bash
tar -czf esmc6b_results.tar.gz \
  runs/5cv_esmc6b_boundary runs/5cv_esmc6b_adapter_only runs/5cv_esmc6b_full \
  logs/ probe_result.json
```

Этот файл некрупный (чекпоинты по ~2-3МБ, 11 штук на ячейку -> весь архив
результатов по всем 60 ячейкам ~1.5-2GB, не 20GB как входные эмбеддинги),
можно смело слать обратно хоть частично готовым, если нужно прерваться раньше.
