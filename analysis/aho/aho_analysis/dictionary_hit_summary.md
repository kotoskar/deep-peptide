# Уточнение AHO: стратификация по тому, СРАБАТЫВАЕТ ли словарь

> _Этот файл автогенерируется скриптом из `src/`; при повторном запуске прозаические подписи вернутся к английскому шаблону (числа — нейтральны)._

Per true TEST peptide, `aho_hit` = the precomputed `pep.inside` AHO feature is nonzero somewhere in the peptide span (some dictionary peptide overlaps it). `hit_source` distinguishes a train(uniprot) hit from an external-AMP-DB-only hit. Recall uplift = AHO models (mean) − baseline, within each bucket.

**Покрытие словарём истинных тестовых пептидов:** 19.8% have a hit. Hit-source breakdown: {'external_only': 207, 'train_only': 19}

| bucket | n | baseline recall | AHO recall (mean) | uplift |
|---|---:|---:|---:|---:|
| dictionary HIT | 226 | 0.668 | 0.748 | +0.080 |
|   ↳ train hit | 19 | 0.842 | 0.895 | +0.053 |
|   ↳ external-only hit | 207 | 0.652 | 0.734 | +0.082 |
| NO hit | 915 | 0.570 | 0.516 | -0.055 |

**Как читать:** бакет без попадания изолирует пептиды, которым словарь помочь не может (нет сигнала) — прирост там должен быть ~0/отрицателен (канал AHO = шум). Бакеты с попаданием показывают, помогает ли AHO там, где срабатывает, и используется ли внешнее попадание (пептид новый для train, но известный AMP-базе) так же, как train-попадание.