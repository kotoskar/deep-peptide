#!/usr/bin/env python3
"""Собирает единую сводную таблицу метрик по ВСЕМ экспериментам.

Источники:
  - analysis/corrected_metrics.csv  — старая (баговая ±3) и новая (исправленная ±3)
    P/R/F1 по all/pep/propep, посчитанные на одном fp32-инференсе.
  - analysis/canonical_metrics.csv  — MCC/AUC (residue-level) + флаг доверия + drift.
  - runs/esmc6b_telescoping/*.json  — новый запуск, ещё не вошедший в canonical.

Логика:
  * СТАРАЯ P/R/F1 = corrected.orig_* (воспроизводит опубликованные train-time значения,
    drift ~0). Для невосстановимых для инференса моделей берём опубликованные значения
    из canonical (train-time), а НОВАЯ = N/A.
  * НОВАЯ P/R/F1 = corrected.corr_* (исправленный матчер ±3).
  * MCC/AUC — единственные (residue-level, у них нет деления старая/новая); берём из
    canonical только там, где test_mcc_auc_trusted, иначе N/A.

Вывод:
  analysis/big_metrics_table.csv  — широкая машинная таблица (всё в одной строке на запуск)
  analysis/big_metrics_table.md   — человекочитаемо: 3 подтаблицы (All / Peptides / Propeptides)
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
A = ROOT / "analysis" / "metrics"

# --- человекочитаемые русские названия запусков ---------------------------------
LABELS = {
    # Таблица 1 (архитектурные изменения, ESM2)
    "train_run_esm2": "ESM2 (базовая)",
    "esm2_telescoping_segmental": "ESM2 + телескопический сегментный CRF",
    "esm2_aho_emission_fusion": "ESM2 + Aho-слияние на эмиссиях",
    "esm2_aho_emission_fusion_h32": "ESM2 + Aho-слияние на эмиссиях (скрытый слой 32)",
    "esm2_aho_mid_fusion_raw_m64": "ESM2 + Aho-слияние на скрытом состоянии",
    "esm2_aho_mid_fusion_raw_m64_pep_only": "ESM2 + Aho-слияние на скрытом состоянии (только пептиды)",
    "esm2_aho_transition_bias_sparse_trainable_zero": "ESM2 + Aho-сигнал в переходы CRF",
    "esm2_aho_tribranch": "ESM2 + Aho раннее слияние (concat)",
    "esm2_bond_loss_soft_l005_w5_tau15": "ESM2 + доп. лосс разрезов к ближайшей границе",
    "train_run_esm2_adamw": "ESM2 с AdamW (60 эпох)",
    # Таблица 2 (генераторы эмбеддингов)
    "train_run_esm2_plus": "ESM2 + биохим. признаки остатков",
    "train_run_esmc_600m": "ESM-C (600M)",
    "esmc_6b": "ESM-C 6B",
    "train_run_prostt5": "ProstT5",
    "train_run_prostt5_plus": "ProstT5 + признаки остатков",
    "train_run_esm2+3di_proj": "(ProstT5-3Di + ESM2) проектор",
    "train_run_esm2+3di_proj_gated": "(ProstT5-3Di + ESM2) проектор + gate",
    "train_run_esm2+3di_proj_gated_conv": "(ProstT5-3Di + ESM2) проектор + gate + conv",
    "train_run_aft": "AFTK все каналы, без фильтра",
    "train_run_aft_single": "AFTK только single, без фильтра",
    "train_run_aft_no_lddt": "AFTK все без lddt, без фильтра",
    "train_run_aft_plddt70": "AFTK все, >70% avg pLDDT",
    "train_run_esm2_aft": "ESM2 + (AFTK все) gate+conv",
    "train_run_esm2_aft_single_gated": "ESM2 + (AFTK single) gate+conv",
    "train_run_esm2_aft_pair_gated": "ESM2 + (AFTK pair) gate+conv",
    "train_run_esm2_aft_no_lddt_gated": "ESM2 + (AFTK без lddt) gate+conv",
    "train_run_esm2_aft_plddt70": "ESM2 + (AFTK >70% pLDDT) gate+conv",
    # Лучшая комбинация + кандидат B
    "esmc6b_boundary_bond": "ESM-C 6B + boundary/bond  ★ лучший",
    "esmc6b_telescoping": "ESM-C 6B + телескопический сегментный CRF",
    # Прочие запуски, не входившие в таблицы Overleaf
    "esm2_boundary_bond_l002_w5_tau15": "ESM2 + boundary/bond",
    "esm2_boundary_only_scale10": "ESM2 + только boundary-голова (scale 10)",
    "esm2_aho_state_bias_pep_boundary_010": "ESM2 + Aho смещение состояний (pep boundary 0.10)",
    "esm2_lora_lstmcnncrf": "ESM2 LoRA (rank8, last4 q/v) + boundary/bond",
    "train_run_esm2_conv": "ESM2 + многомасштабный проектор (multiscale)",
    "train_run_esm2_only_homo": "ESM2 (обучение только на Homo, 40 эпох)",
    "train_run_esm2_plus_proj_gated": "ESM2+ + трёхветочный gated-проектор",
    "uni2026_run_esm2": "ESM2 на датасете uniprot_2026",
    "scaling_trainfrac50": "Масштабирование данных: 50% train",
    "scaling_trainfrac60": "Масштабирование данных: 60% train",
    "scaling_trainfrac70": "Масштабирование данных: 70% train",
    "scaling_trainfrac80": "Масштабирование данных: 80% train",
    "scaling_trainfrac90": "Масштабирование данных: 90% train",
    "train_run_esm2_100": "Масштабирование данных: 100% train",
    "train_run_esm2_25": "Масштабирование данных: 25% train (старая серия, ненадёжно)",
    "train_run_esm2_50": "Масштабирование данных: 50% train (старая серия)",
    "train_run_esm2_75": "Масштабирование данных: 75% train (старая серия)",
}

UNRECOVERABLE = {
    "esm2_bond_loss_soft_l005_w5_tau15",
    "esm2_aho_transition_bias_sparse_trainable_zero",
}


def load_csv(path):
    with open(path) as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def f(x, nd=3):
    """Форматирует число или N/A."""
    if x is None or x == "" or x == "N/A":
        return "N/A"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "N/A"


def add_telescoping(canon):
    """Добавляет esmc6b_telescoping в canonical-словарь из его JSON-файлов."""
    rd = ROOT / "runs" / "esmc6b_telescoping"
    tm = json.load(open(rd / "test_metrics.json"))
    inf = json.load(open(rd / "test_metrics_infer.json"))
    row = {"run": "esmc6b_telescoping", "test_mcc_auc_trusted": "True", "test_drift": "0.0"}
    for cls in ("all", "peptides", "propeptides"):
        row[f"f1_{cls}"] = tm[f"f1 {cls}"]
        row[f"precision_{cls}"] = tm[f"precision {cls}"]
        row[f"recall_{cls}"] = tm[f"recall {cls}"]
        row[f"mcc_{cls}"] = inf[f"residue mcc {cls}"]
        row[f"auc_{cls}"] = inf[f"residue roc_auc {cls}"]
    canon["esmc6b_telescoping"] = row


def main():
    corr = load_csv(A / "corrected_metrics.csv")
    canon = load_csv(A / "canonical_metrics.csv")
    if "esmc6b_telescoping" not in canon:
        add_telescoping(canon)

    runs = sorted(set(corr) | set(canon), key=lambda r: LABELS.get(r, r))
    wide = []
    for run in runs:
        c = corr.get(run, {})
        k = canon.get(run, {})
        trusted = str(k.get("test_mcc_auc_trusted", "")).lower() == "true"
        row = {"run": run, "label": LABELS.get(run, run)}
        for cls in ("all", "peptides", "propeptides"):
            # СТАРАЯ P/R/F1: corrected.orig_*, иначе train-time published из canonical
            if c:
                row[f"old_f1_{cls}"] = c.get(f"orig_{cls}_f1")
                row[f"old_p_{cls}"] = c.get(f"orig_{cls}_precision")
                row[f"old_r_{cls}"] = c.get(f"orig_{cls}_recall")
                row[f"new_f1_{cls}"] = c.get(f"corr_{cls}_f1")
                row[f"new_p_{cls}"] = c.get(f"corr_{cls}_precision")
                row[f"new_r_{cls}"] = c.get(f"corr_{cls}_recall")
            else:  # невосстановимая модель — только published старая, новой нет
                row[f"old_f1_{cls}"] = k.get(f"f1_{cls}")
                row[f"old_p_{cls}"] = k.get(f"precision_{cls}")
                row[f"old_r_{cls}"] = k.get(f"recall_{cls}")
                row[f"new_f1_{cls}"] = row[f"new_p_{cls}"] = row[f"new_r_{cls}"] = "N/A"
            # MCC/AUC: residue-level, единственные, только если доверяем
            row[f"mcc_{cls}"] = k.get(f"mcc_{cls}") if trusted else "N/A"
            row[f"auc_{cls}"] = k.get(f"auc_{cls}") if trusted else "N/A"
        wide.append(row)

    # --- CSV ---
    cols = ["run", "label"]
    for cls in ("all", "peptides", "propeptides"):
        for m in ("old_f1", "new_f1", "old_p", "new_p", "old_r", "new_r", "mcc", "auc"):
            cols.append(f"{m}_{cls}")
    with open(A / "big_metrics_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(wide)

    # --- Markdown: 3 подтаблицы ---
    def subtable(cls, title):
        L = [f"### {title}", "",
             "| Эксперимент | F1 стар. | F1 нов. | P стар. | P нов. | R стар. | R нов. | MCC | AUC |",
             "|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in wide:
            L.append("| {lab} | {f1o} | {f1n} | {po} | {pn} | {ro} | {rn} | {mcc} | {auc} |".format(
                lab=r["label"],
                f1o=f(r[f"old_f1_{cls}"]), f1n=f(r[f"new_f1_{cls}"]),
                po=f(r[f"old_p_{cls}"]), pn=f(r[f"new_p_{cls}"]),
                ro=f(r[f"old_r_{cls}"]), rn=f(r[f"new_r_{cls}"]),
                mcc=f(r[f"mcc_{cls}"]), auc=f(r[f"auc_{cls}"])))
        return "\n".join(L)

    n_old_only = sum(1 for r in wide if r["run"] in UNRECOVERABLE)
    header = f"""# Единая сводная таблица метрик по всем экспериментам

> **Как читать.** Метрика поиска пептидов с допуском ±3 по разрезам считается двумя
> способами: **стар.** — как в оригинальном DeepPeptide (с багом затенения переменной,
> занижает recall на ~2–4 п.п.; значения сопоставимы со статьёй) и **нов.** —
> исправленный матчер. Деление «стар./нов.» относится ТОЛЬКО к P/R/F1 (баг живёт в
> сегментном матчере). **MCC и AUC** считаются на уровне остатков (residue-level) и
> единственны — у них нет версий «стар./нов.». MCC/AUC берутся из свежего fp32-инференса
> и приводятся только там, где он воспроизводит train-time P/R/F1 (drift ≤ 0.015); иначе N/A.
>
> Методику расчёта см. в `texs/error_analysis/methodology.md`.

Всего запусков в таблице: **{len(wide)}**. Из них {n_old_only} с реальными старыми
P/R/F1, но невосстановимой для инференса моделью → у них новая P/R/F1 и MCC/AUC = N/A
(строки сохранены, см. сноску). Строки без запуска в `runs/` (например исторический
`(ProstT5 3DI + ESM2+) proj.gated.conv.`) в таблицу не включены — соответствующего
эксперимента нет.

★ — лучший результат проекта (ESM-C 6B + boundary/bond).
"""
    foot = """
---

**Сноска — невосстановимые для инференса модели** (`esm2_bond_loss_soft_l005_w5_tau15`,
`esm2_aho_transition_bias_sparse_trainable_zero`): класс модели отсутствует в текущем
коде, чекпойнт не загружается. Старые P/R/F1 — это train-time published значения;
новая P/R/F1 и MCC/AUC недоступны (N/A).
"""
    md = "\n\n".join([header,
                      subtable("all", "All (пептиды + пропептиды вместе)"),
                      subtable("peptides", "Peptides"),
                      subtable("propeptides", "Propeptides"),
                      foot])
    (A / "big_metrics_table.md").write_text(md)
    print(f"Готово: {len(wide)} запусков → big_metrics_table.{{csv,md}}")
    print(f"  невосстановимых (только старая P/R/F1): {n_old_only}")


if __name__ == "__main__":
    main()
