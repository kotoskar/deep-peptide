#!/usr/bin/env python3
"""Новизна инкрементального содержимого ревизии 2026 относительно того, на чём мы
обучались (2022-TRAIN).

Зачем (задача A): 2026 ≈ 2022 + ~1178 новых белков (8441 общих ID, 8 удалено). Вопрос:
эти новые данные приносят НОВЫЕ пептидные семейства (тогда обучение на 2026 расширит
покрытие — наш реальный bottleneck, см. data_need.md) или это те же семейства +объём
(тогда эффект ~как +15% данных на кривой масштабирования, ~+0.02 F1)?

Почему НЕ «2026-test vs 2022-train» в лоб: сплиты 2026 и 2022 считались независимо, и
белок мог мигрировать 2022-train → 2026-test между ревизиями (явный пример из
dataset_stats: Cyriopagopus 62/115/81 в 2022, но 266/0/0 в 2026). Тогда «2026-test»
содержит белки, бывшие в 2022-train → ложная не-новизна. Чистая реализация замысла —
**пептиды ТОЛЬКО-новых белков (1178) против 2022-train**.

Контроль шумового пола (важно — 2026 не препроцессирован как 2022, нет обработки
фланкирующих мотивов → иная обрезка границ может дать <100% идентичность для
биологически ТОГО ЖЕ пептида): извлекаем пептиды тех же 2022-train белков, но из ФАЙЛА
2026, и выравниваем против 2022-train DB. Идентичность должна быть ~1.0; зазор = пол
шума от разной обрезки. На него дисконтируем «новизну». Поэтому печатаем И identity,
И coverage (matches/min_len) — coverage ловит случай «тот же пептид, иначе обрезан».

Вывод: analysis/similarity/novelty_2026.csv + дописывает раздел в summary.md.
Эмбеддинги НЕ нужны — только последовательности + needleall.

Usage: env/bin/python analysis/similarity/src/novelty_2026_vs_2022train.py
"""
from __future__ import annotations
import sys, tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())))
from src.utils.crf_label_utils import parse_coordinate_string
# переиспользуем выравниватель из соседнего скрипта
from analysis.similarity.src.peptide_similarity import needleall_max_identity, species

D22 = "data/uniprot_2022/labeled_sequences.csv"
G22 = "data/uniprot_2022/graphpart_assignments.csv"
D26 = "data/uniprot_2026/labeled_sequences.csv"
SPLIT22 = {0: "train", 1: "train", 2: "train", 3: "valid", 4: "test"}
OUT = Path("analysis/similarity")
MIN_LEN, MAX_LEN = 5, 100
SIM = 0.70


def peptides_of(df_rows):
    """rows -> {type: {seq: count}} (merged-overlap segments, len-filtered)."""
    out = {"pep": defaultdict(int), "propep": defaultdict(int)}
    for _, row in df_rows.iterrows():
        seq = row["sequence"]
        for col, typ in [("coordinates", "pep"), ("propeptide_coordinates", "propep")]:
            if pd.isna(row[col]):
                continue
            for st, en in parse_coordinate_string(str(row[col]), merge_overlaps=True):
                pep = seq[st - 1:en]
                if MIN_LEN <= len(pep) <= MAX_LEN:
                    out[typ][pep] += 1
    return out


def stats(res):
    """res: {seq:(ident,cov,best)} -> dict of distribution stats."""
    if not res:
        return None
    idents = sorted(v[0] for v in res.values())
    covs = sorted(v[1] for v in res.values())
    n = len(idents)
    med = idents[n // 2]
    lt30 = sum(i < 0.30 for i in idents) / n
    ge70 = sum(i >= SIM for i in idents) / n
    cov_ge70 = sum(c >= SIM for c in covs) / n
    return {"n": n, "median_id": med, "frac_lt30": lt30,
            "frac_ge70_id": ge70, "frac_ge70_cov": cov_ge70}


def main():
    df22 = pd.read_csv(D22, index_col=0)
    gp22 = pd.read_csv(G22, index_col="AC")
    df26 = pd.read_csv(D26, index_col=0)

    ids22 = set(df22["protein_id"])
    # 2022-train protein ids
    train_ids = {ac for ac in gp22.index if SPLIT22.get(int(gp22.loc[ac, "cluster"])) == "train"}
    train_ids &= set(df22["protein_id"])

    # DB = 2022-train unique peptides (the data the model was trained on)
    db = peptides_of(df22[df22["protein_id"].isin(train_ids)])

    # query A: peptides of NEW-only 2026 proteins (genuinely incremental content)
    new_only = df26[~df26["protein_id"].isin(ids22)]
    q_new = peptides_of(new_only)

    # control: peptides of SHARED 2022-train proteins as they appear in the 2026 FILE
    #          (same proteins, possibly re-trimmed) -> boundary-noise floor (expect ~1.0)
    shared_train_26 = df26[df26["protein_id"].isin(train_ids)]
    q_floor = peptides_of(shared_train_26)

    print(f"new-only 2026 proteins: {len(new_only)}")
    for t in ("pep", "propep"):
        print(f"  {t}: DB(train)={len(db[t])}  new={len(q_new[t])}  floor={len(q_floor[t])}")

    rows = []
    results = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for typ in ("pep", "propep"):
            db_seqs = list(db[typ].keys())
            if not db_seqs:
                continue
            for tag, q in [("new_only", q_new[typ]), ("floor_train26", q_floor[typ])]:
                q_seqs = list(q.keys())
                if not q_seqs:
                    continue
                wd = td / f"{tag}_{typ}"; wd.mkdir()
                print(f"aligning {tag}/{typ}: {len(q_seqs)} x {len(db_seqs)} ...", flush=True)
                res = needleall_max_identity(q_seqs, db_seqs, wd)
                results[(tag, typ)] = stats(res)
                for seq, (ident, cov, best) in res.items():
                    rows.append({"seq": seq, "type": typ, "set": tag, "length": len(seq),
                                 "n_occurrences": q[seq],
                                 "max_identity_to_2022train": round(ident, 4),
                                 "coverage_at_best": round(cov, 4),
                                 "is_similar_70": ident >= SIM})
    pd.DataFrame(rows).to_csv(OUT / "novelty_2026.csv", index=False)
    write_summary(results, len(new_only))
    print(f"\nWrote {OUT}/novelty_2026.csv ({len(rows)} rows)")


def write_summary(results, n_new_proteins):
    L = ["", "---", "",
         "## Новизна ревизии 2026 относительно 2022-TRAIN (задача A)", "",
         f"2026 ≈ 2022 + **{n_new_proteins} только-новых белков** (8441 общих ID, 8 удалено). "
         "Измеряем макс. идентичность пептидов к 2022-train (то, на чём обучались). "
         "`floor_train26` = контроль: те же 2022-train белки, извлечённые из файла 2026 "
         "(пол шума от иной обрезки границ; ожидаем ~1.0). identity = matches/aln_len; "
         "coverage = matches/min(len). Воспроизв.: `analysis/similarity/src/novelty_2026_vs_2022train.py`.", ""]
    L += ["| набор / тип | n уник | медиана id | доля <0.30 | доля ≥70% id | доля ≥70% cov |",
          "|---|---:|---:|---:|---:|---:|"]
    name = {"new_only": "НОВЫЕ белки 2026", "floor_train26": "контроль (train из файла 2026)"}
    for tag in ("new_only", "floor_train26"):
        for typ in ("pep", "propep"):
            s = results.get((tag, typ))
            if not s:
                continue
            L.append(f"| {name[tag]} / {typ} | {s['n']} | {s['median_id']:.2f} | "
                     f"{s['frac_lt30']:.0%} | {s['frac_ge70_id']:.0%} | {s['frac_ge70_cov']:.0%} |")
    L += ["",
          "**Как читать.** Если у контроля (`floor`) доля ≥70% сильно ниже 100% — значит "
          "разная обрезка границ в 2026 сама по себе создаёт «псевдо-новизну»; на этот "
          "зазор дисконтируем строку НОВЫХ белков. Тяжёлый хвост <0.30 у новых белков "
          "(сверх пола) = новые семейства → обучение на 2026 расширит покрытие "
          "(наш bottleneck). Преобладание ≥0.70 = те же семейства +объём → эффект ~как "
          "+15% данных (~+0.02 F1 по кривой масштабирования). Это отвечает на «ново ли "
          "инкрементальное содержимое», а НЕ «поднимет ли 2026 метрики» (для последнего "
          "нужен валидный сплит 2026 + переобучение)."]
    p = OUT / "summary.md"
    p.write_text(p.read_text() + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
