#!/usr/bin/env python3
"""Острый срез к задаче A: закрывает ли инкрементальное содержимое 2026 именно те
тестовые пептиды 2022, которые модель СЕЙЧАС проваливает?

Наш bottleneck (data_need.md): 2022-test пептиды с <0.30 идентичности к 2022-train
распознаются с recall 0.39. Вопрос: если добавить 1178 новых белков 2026 в обучение,
появится ли у этих трудных тест-пептидов близкий (≥70%) сосед среди новых? Доля
«закрытых» = насколько 2026 чинит ровно нашу дыру (а не приносит абстрактную новизну).

Метод: берём 2022-test пептиды с max_identity_to_train < 0.30 из
peptide_similarity.csv; выравниваем их против пептидов 1178 только-новых белков 2026
(novelty_2026.csv, set=new_only). Per type. Печатаем долю с ≥70% identity / coverage.

Usage: env/bin/python analysis/similarity/src/gapfill_2026.py
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())))
from analysis.similarity.src.peptide_similarity import needleall_max_identity

OUT = Path("analysis/similarity")
SIM = 0.70
HARD = 0.30


def main():
    sim = pd.read_csv(OUT / "peptide_similarity.csv")
    nov = pd.read_csv(OUT / "novelty_2026.csv")
    type_map = {"pep": "pep", "propep": "propep"}
    lines = ["", "### Острый срез: закрывает ли 2026 трудные (<0.30) тест-пептиды 2022?", "",
             f"2022-test пептиды с идентичностью к 2022-train < {HARD:.2f} (это наш bottleneck, "
             f"recall≈0.39), выровненные против пептидов 1178 только-новых белков 2026. "
             "Доля ≥70% = сколько наших трудных пептидов получили бы близкого соседа в обучении.", "",
             "| тип | трудных тест-пептидов (<0.30) | покрыто ≥70% id | покрыто ≥70% cov |",
             "|---|---:|---:|---:|"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for typ in ("pep", "propep"):
            hard = sim[(sim["split"] == "test") & (sim["type"] == typ) &
                       (sim["max_identity_to_train"] < HARD)]["seq"].tolist()
            newp = nov[(nov["set"] == "new_only") & (nov["type"] == typ)]["seq"].tolist()
            if not hard or not newp:
                continue
            wd = td / typ; wd.mkdir()
            res = needleall_max_identity(hard, newp, wd)
            ge70_id = sum(v[0] >= SIM for v in res.values())
            ge70_cov = sum(v[1] >= SIM for v in res.values())
            n = len(res)
            print(f"{typ}: {n} hard test peptides; ≥70% id covered by new-2026: {ge70_id} ({100*ge70_id/n:.0f}%)")
            lines.append(f"| {typ} | {n} | {ge70_id} ({100*ge70_id/n:.0f}%) | {ge70_cov} ({100*ge70_cov/n:.0f}%) |")
    lines += ["",
              "**Вывод.** Высокая доля = 2026 приносит соседей именно для пептидов, которые мы "
              "проваливаем → целевое закрытие дыры. Низкая = новые белки новые, но НЕ те семейства, "
              "что нам нужны → 2026 поможет лишь как общий +объём, а адресный bottleneck остаётся "
              "за идеей (b) (затащить недопокрытые семейства из AMP-баз)."]
    p = OUT / "summary.md"
    p.write_text(p.read_text() + "\n".join(lines) + "\n")
    print("appended to summary.md")


if __name__ == "__main__":
    main()
