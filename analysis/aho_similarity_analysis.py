#!/usr/bin/env python3
"""Does the AHO prior help more on test peptides SIMILAR to train vs NOVEL ones?

The AHO prior is an Aho-Corasick dictionary of known bioactive peptides built
fold-aware from TRAIN folds (0,1,2) + external AMP databases — so it can only
"fire" on a test peptide whose (sub)sequence resembles something already known.
Supervisor's question: quantify the AHO benefit on test peptides ≥70% identical
to a train peptide vs novel (<70%) ones.

Method: run TEST inference for a no-AHO baseline and AHO-fusion model(s); for each
true peptide segment record whether it was recovered (±3 matching) and its
sequence; join the sequence to its max-identity-to-train bucket
(`analysis/peptide_similarity/peptide_similarity.csv`); compare per-bucket recall
of AHO vs baseline.

Output: analysis/aho_analysis/ (csv + summary.md + figure).
Usage: env/bin/python analysis/aho_similarity_analysis.py [--device 0]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.error_analysis import run_inference, _group
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders, PEPTIDE_START_STATE, PEPTIDE_END_STATE,
)

OUT = Path("analysis/aho_analysis"); OUT.mkdir(parents=True, exist_ok=True)
BASELINE = "train_run_esm2"
AHO_MODELS = ["esm2_aho_emission_fusion", "esm2_aho_emission_fusion_h32",
              "esm2_aho_mid_fusion_raw_m64"]
SIM_THRESHOLD = 0.70


def matched_true_segments(run: str, device: str):
    """Return list of dicts {protein_id, seq, length, matched} for true PEPTIDE segments."""
    preds, names, data, _ = run_inference(Path("runs") / run, device)
    out = []
    for i, pred in enumerate(preds):
        pid = names[i]; row = data.loc[pid]; protein_seq = row["sequence"]
        true_segs = [(int(a), int(b)) for a, b in row["true_peptides"]]
        pred_segs = convert_path_to_peptide_borders(
            pred, start_state=PEPTIDE_START_STATE, stop_state=PEPTIDE_END_STATE, offset=1)
        for group in _group(true_segs):
            # representative = longest member
            s, e = max(group, key=lambda z: z[1] - z[0])
            seq = protein_seq[s - 1:e]
            matched = any(
                abs(ps - ts) <= 3 and abs(pe - te) <= 3
                for (ts, te) in group for (ps, pe) in pred_segs)
            out.append({"protein_id": pid, "seq": seq, "length": e - s + 1, "matched": matched})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if __import__("torch").cuda.is_available() else "cpu"

    sim = pd.read_csv("analysis/peptide_similarity/peptide_similarity.csv")
    sim = sim[(sim["split"] == "test") & (sim["type"] == "pep")][["seq", "max_identity_to_train"]]
    sim_map = dict(zip(sim["seq"], sim["max_identity_to_train"]))

    frames = {}
    print(f"[baseline] {BASELINE}")
    frames["baseline"] = matched_true_segments(BASELINE, device)
    for m in AHO_MODELS:
        if not (Path("runs") / m / "model.pt").exists():
            print(f"[skip] {m}"); continue
        print(f"[aho] {m}")
        frames[m] = matched_true_segments(m, device)

    # attach similarity + bucket (join by sequence; segments not in the unique-sim
    # table get NaN -> dropped from the stratified view but kept in overall)
    rows = []
    for label, df in frames.items():
        df = df.copy()
        df["model"] = label
        df["max_id"] = df["seq"].map(sim_map)
        df["bucket"] = np.where(df["max_id"].isna(), "unknown",
                        np.where(df["max_id"] >= SIM_THRESHOLD, f"similar(≥{SIM_THRESHOLD:.0%})", "novel(<70%)"))
        rows.append(df)
    big = pd.concat(rows, ignore_index=True)
    big.to_csv(OUT / "aho_segments.csv", index=False)

    # pooled AHO = mean over AHO models per (seq) — but simpler: report each + a pooled recall
    aho_labels = [l for l in frames if l != "baseline"]
    write_summary(big, aho_labels)
    print(f"\nWrote {OUT}/summary.md")


def _recall(df):
    return df["matched"].mean() if len(df) else float("nan")


def write_summary(big, aho_labels):
    lines = ["# AHO prior: benefit on similar-to-train vs novel test peptides", ""]
    lines.append("Recall (±3) on true TEST **peptides**, stratified by max identity to any "
                 f"train peptide. similar = ≥{SIM_THRESHOLD:.0%} identity, novel = <70%. "
                 "AHO dictionary is fold-aware (train folds + external AMP DBs), so it can "
                 "only match peptides resembling known ones.")
    lines.append("")
    base = big[big["model"] == "baseline"]
    # bucket sizes (same true peptides regardless of model)
    bsz = base["bucket"].value_counts().to_dict()
    lines.append(f"Bucket sizes (true test peptides): {bsz}")
    lines.append("")
    lines.append("| model | recall novel(<70%) | recall similar(≥70%) | recall all | Δ(sim−novel) |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in ["baseline"] + aho_labels:
        d = big[big["model"] == label]
        rn = _recall(d[d["bucket"] == "novel(<70%)"])
        rs = _recall(d[d["bucket"] == f"similar(≥{SIM_THRESHOLD:.0%})"])
        ra = _recall(d)
        lines.append(f"| {label} | {rn:.3f} | {rs:.3f} | {ra:.3f} | {rs-rn:+.3f} |")
    lines.append("")
    # AHO uplift vs baseline per bucket (pooled AHO = average of AHO models' recall)
    lines.append("## AHO uplift over baseline, per similarity bucket")
    lines.append("")
    lines.append("| bucket | baseline recall | AHO recall (mean of models) | uplift |")
    lines.append("|---|---:|---:|---:|")
    for bucket in ["novel(<70%)", f"similar(≥{SIM_THRESHOLD:.0%})"]:
        rb = _recall(base[base["bucket"] == bucket])
        aho_recs = [_recall(big[(big["model"] == m) & (big["bucket"] == bucket)]) for m in aho_labels]
        ra = float(np.nanmean(aho_recs)) if aho_recs else float("nan")
        lines.append(f"| {bucket} | {rb:.3f} | {ra:.3f} | {ra-rb:+.3f} |")
    lines.append("")
    lines.append("**Reading:** if the AHO uplift is concentrated in the similar(≥70%) bucket "
                 "and ~0 (or negative) on novel peptides, the AHO prior works by retrieving "
                 "known peptides rather than improving genuine generalization — quantifying the "
                 "supervisor's concern.")
    (OUT / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
