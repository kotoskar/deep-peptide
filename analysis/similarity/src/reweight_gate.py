#!/usr/bin/env python3
"""Аналитический гейт перед обучающим экспериментом C (reweighting по редкости семейства).

Логика (симметрична задаче A): oversampling редких train-семейств может ВЫТАЩИТЬ соседа,
только если он СУЩЕСТВУЕТ в train. Для хвоста <0.30 соседа нет (показано в A) → reweighting
там бессилен. Реальный шанс C — СРЕДНИЕ бакеты 0.30–0.70, где у тест-пептида ЕСТЬ train-сосед,
но его семейство редкое и «тонет» при обычном shuffle.

Гейт: предсказывает ли ПЛОТНОСТЬ train-семейства соседа recall тест-пептида ПРИ ФИКСИРОВАННОЙ
идентичности? Если у тест-пептидов с РЕДКИМ соседом (низкая плотность) recall заметно ниже,
чем с частым — значит редкие семейства недообучены → у reweighting есть headroom. Если recall
плоский по плотности — reweighting не поможет, и 6ч обучения тратить не стоит.

Шаги:
  1) train-vs-train needleall (пептиды): density[train_pep] = число ДРУГИХ train-пептидов
     с идентичностью ≥0.70 (self исключаем).
  2) тест-пептиды (peptide_similarity.csv): best_train_seq + max_identity_to_train.
  3) плотность семейства соседа = density[best_train_seq].
  4) recall тест-пептида = matched из aho_segments.csv (model=baseline).
  5) в среднем бакете 0.30–0.70 сравниваем recall при низкой vs высокой плотности соседа.

Без GPU. Usage: env/bin/python analysis/similarity/src/reweight_gate.py
"""
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())))
from analysis.similarity.src.peptide_similarity import write_fasta

OUT = Path("analysis/similarity")
SIM = 0.70


def needleall_neighbor_count(seqs, workdir, thr=SIM):
    """Для каждой seq: число ДРУГИХ seq с идентичностью ≥ thr (self исключён)."""
    smap = write_fasta(seqs, workdir / "s.fa")
    cnt = {s: 0 for s in seqs}
    inv = {f"q{i}": s for i, s in enumerate(seqs)}
    proc = subprocess.Popen(
        ["needleall", "-auto", "-asequence", str(workdir / "s.fa"),
         "-bsequence", str(workdir / "s.fa"), "-gapopen", "10", "-gapextend", "0.5",
         "-aformat3", "pair", "-errfile", str(workdir / "err.txt"), "-outfile", "stdout"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    cur_q = cur_d = None
    for line in proc.stdout:
        if line.startswith("# 1:"):
            cur_q = line.split(":", 1)[1].strip()
        elif line.startswith("# 2:"):
            cur_d = line.split(":", 1)[1].strip()
        elif line.startswith("# Identity:"):
            x, y = line.split(":", 1)[1].strip().split("(")[0].strip().split("/")
            ident = int(x) / int(y) if int(y) else 0.0
            if ident >= thr and cur_q != cur_d:  # исключаем self-hit
                cnt[inv[cur_q]] += 1
    proc.wait()
    return cnt


def main():
    sim = pd.read_csv(OUT / "peptide_similarity.csv")
    seg = pd.read_csv("analysis/aho/aho_analysis/aho_segments.csv")
    seg = seg[seg["model"] == "baseline"][["seq", "matched"]].drop_duplicates("seq")

    # train-пептиды = best_train_seq, встречающиеся у тест/valid пептидов (тип pep)
    sim_pep = sim[sim["type"] == "pep"].copy()
    train_seqs = sorted(set(sim_pep["best_train_seq"].dropna()))
    print(f"train pep neighbours to score for density: {len(train_seqs)}")
    with tempfile.TemporaryDirectory() as td:
        dens = needleall_neighbor_count(train_seqs, Path(td), thr=SIM)

    test = sim_pep[sim_pep["split"] == "test"].copy()
    test["neighbor_density"] = test["best_train_seq"].map(dens)
    test = test.merge(seg, on="seq", how="inner")  # add baseline matched
    print(f"test pep joined with recall: {len(test)}")

    mid = test[(test["max_identity_to_train"] >= 0.30) & (test["max_identity_to_train"] < 0.70)].copy()
    med = mid["neighbor_density"].median()
    lo = mid[mid["neighbor_density"] <= med]
    hi = mid[mid["neighbor_density"] > med]

    def rec(df):
        return df["matched"].mean() if len(df) else float("nan")

    L = ["", "---", "",
         "### Гейт для C (reweighting по редкости семейства)", "",
         "Вопрос: предсказывает ли плотность train-семейства соседа recall тест-пептида при "
         "фиксированной идентичности? Если у пептидов с РЕДКИМ соседом recall ниже — у "
         "reweighting есть headroom; если плоско — нет. Только пептиды, baseline ESM2. "
         "Плотность = число train-пептидов ≥70% идентичных соседу (self исключён). "
         "Воспроизв.: `analysis/similarity/src/reweight_gate.py`.", "",
         f"Средний бакет (0.30 ≤ identity < 0.70): n={len(mid)}, медиана плотности соседа={med:.0f}.", "",
         "| подгруппа среднего бакета | n | recall (baseline) |",
         "|---|---:|---:|",
         f"| РЕДКИЙ сосед (density ≤ {med:.0f}) | {len(lo)} | {rec(lo):.3f} |",
         f"| ЧАСТЫЙ сосед (density > {med:.0f}) | {len(hi)} | {rec(hi):.3f} |",
         "",
         f"Для контекста — recall по всему среднему бакету: {rec(mid):.3f}; "
         f"<0.30 (хвост, reweighting бессилен): {rec(test[test['max_identity_to_train']<0.30]):.3f}; "
         f"≥0.70: {rec(test[test['max_identity_to_train']>=0.70]):.3f}.", "",
         f"**Вывод гейта.** Разрыв recall (редкий−частый) = {rec(lo)-rec(hi):+.3f}. "
         "Заметно отрицательный (редкие < частых) → редкие семейства недообучены, у reweighting "
         "есть headroom → запускать C. Около нуля → reweighting вряд ли поможет (recall в среднем "
         "бакете определяется не редкостью, а самой умеренной похожестью) → C низкоприоритетен, "
         "лучше идея (b)."]
    p = OUT / "summary.md"
    p.write_text(p.read_text() + "\n".join(L) + "\n")
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
