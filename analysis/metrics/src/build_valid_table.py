#!/usr/bin/env python3
"""Build the VALIDATION metrics table (analysis/metrics/valid_big_table.{csv,md}).

Methodological companion to the test big table: model selection should be judged on
the validation partition, not test. Reads each run's valid_metrics_infer.json (written
by analysis/metrics/src/infer_valid.py) and emits a table ranked by val F1 (all), with
the residue MCC/AUC and — for the overfitting check — the run's TEST F1 and Δ=val−test.

NOTE: val F1 is the SELECTION metric (best checkpoint was chosen on val), so it is
optimistic vs test; the point is the RANKING consistency, not absolute values. Runs on
small/filtered val sets (e.g. *_plddt70) have noisy val numbers — read with care.

Run from repo root:
  env/bin/python analysis/metrics/src/build_valid_table.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
A = ROOT / "analysis" / "metrics"
import sys
sys.path.insert(0, str(A / "src"))
from build_big_table import LABELS  # reuse the human-readable run names


def jload(p):
    return json.load(open(p)) if p.exists() else None


def main():
    rows = []
    for d in sorted((ROOT / "runs").iterdir()):
        vj = jload(d / "valid_metrics_infer.json")
        if not vj:
            continue
        tj = jload(d / "test_metrics_infer.json") or {}
        row = {
            "run": d.name,
            "label": LABELS.get(d.name, d.name),
            "val_f1_all": vj.get("f1 all"),
            "val_p_all": vj.get("precision all"),
            "val_r_all": vj.get("recall all"),
            "val_mcc_all": vj.get("residue mcc all"),
            "val_auc_all": vj.get("residue roc_auc all"),
            "val_f1_pep": vj.get("f1 peptides"),
            "val_f1_propep": vj.get("f1 propeptides"),
            "test_f1_all": tj.get("f1 all"),
        }
        if row["val_f1_all"] is not None and row["test_f1_all"] is not None:
            row["val_minus_test_f1"] = row["val_f1_all"] - row["test_f1_all"]
        else:
            row["val_minus_test_f1"] = None
        rows.append(row)

    rows.sort(key=lambda r: (r["val_f1_all"] is not None, r["val_f1_all"] or 0), reverse=True)
    cols = ["run", "label", "val_f1_all", "val_p_all", "val_r_all", "val_mcc_all",
            "val_auc_all", "val_f1_pep", "val_f1_propep", "test_f1_all", "val_minus_test_f1"]
    with open(A / "valid_big_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/A"

    L = ["# Валидационная сводная таблица (метрика селекции)", "",
         "> Методически модель выбирается по **валиду** (кластер 3), не по тесту. Здесь — "
         "те же запуски, ранжированные по val F1 (all), + residue MCC/AUC и, для проверки "
         "переобучения, тестовый F1 и Δ=val−test. **val F1 — метрика селекции** (по ней "
         "выбран лучший чекпойнт), поэтому она оптимистична относительно теста; смотреть надо "
         "на согласованность РАНГА. Запуски с маленьким/фильтрованным валидом (`*_plddt70`, "
         "lora) дают шумные val-числа.", "",
         "| Эксперимент | val F1 | val P | val R | val MCC | val AUC | val F1 пеп | val F1 пропеп | test F1 | Δ(val−test) |",
         "|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        L.append("| {lab} | {f1} | {p} | {rc} | {mcc} | {auc} | {fp} | {fpr} | {tf} | {d} |".format(
            lab=r["label"], f1=fmt(r["val_f1_all"]), p=fmt(r["val_p_all"]), rc=fmt(r["val_r_all"]),
            mcc=fmt(r["val_mcc_all"]), auc=fmt(r["val_auc_all"]), fp=fmt(r["val_f1_pep"]),
            fpr=fmt(r["val_f1_propep"]), tf=fmt(r["test_f1_all"]), d=fmt(r["val_minus_test_f1"])))
    (A / "valid_big_table.md").write_text("\n".join(L) + "\n")
    print(f"wrote {A/'valid_big_table.csv'} and .md ({len(rows)} runs)")


if __name__ == "__main__":
    main()
