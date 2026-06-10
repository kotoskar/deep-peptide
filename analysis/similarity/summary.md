# Похожесть held-out пептидов к TRAIN на уровне пептидов

> _Этот файл автогенерируется скриптом из `src/`; при повторном запуске прозаические подписи вернутся к английскому шаблону (числа — нейтральны)._

На каждый уникальный valid/test пептид: макс. идентичность (needle, matches/alignment_length) к любому train-пептиду того же типа. `is_similar_70` = идентичность ≥ 70%. `coverage_at_best` = matches/min(len) на лучшем попадании (ловит вложение короткого в длинный).

> Сплит разделён по гомологии на уровне ЦЕЛОГО БЕЛКА (GraphPart needle, 30%). Здесь измеряется, новы ли и сами СЕГМЕНТЫ пептидов.

- **valid/pep** (n=747 unique): median max-identity 0.34; **5% ≥70% identity** to a train peptide; 9% ≥70% coverage (containment).
- **valid/propep** (n=1068 unique): median max-identity 0.36; **1% ≥70% identity** to a train peptide; 4% ≥70% coverage (containment).
- **test/pep** (n=852 unique): median max-identity 0.37; **6% ≥70% identity** to a train peptide; 8% ≥70% coverage (containment).
- **test/propep** (n=1118 unique): median max-identity 0.35; **0% ≥70% identity** to a train peptide; 2% ≥70% coverage (containment).

## Распределение макс.-идентичности-к-train (test, оба типа)

| бин идентичности | пеп | пропеп |
|---|---:|---:|
| <0.30 | 113 | 149 |
| 0.30–0.50 | 568 | 896 |
| 0.50–0.70 | 124 | 70 |
| 0.70–0.90 | 33 | 3 |
| 0.90–1.00 | 14 | 0 |
