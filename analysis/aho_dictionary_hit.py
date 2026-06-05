#!/usr/bin/env python3
"""Refinement of the AHO analysis: stratify by whether the dictionary ACTUALLY fires.

The earlier AHO analysis bucketed test peptides by similarity to TRAIN. But the AHO
dictionary also contains ~41k external AMP-DB peptides, so a peptide that is novel
*to train* could still be a dictionary hit via an external DB. This script buckets
each true test peptide by whether the precomputed AHO feature `pep.inside` actually
fires anywhere in the peptide span (i.e. some dictionary peptide overlaps it), and
further by the hit SOURCE (train uniprot vs external-only), then compares the
recall uplift of AHO models over the no-AHO baseline within each bucket.

Reuses the matched flags already computed in analysis/aho_analysis/aho_segments.csv
(no re-inference); reads dictionary hits from data/embeddings_aho_train012.

Output: analysis/aho_analysis/dictionary_hit_summary.md (+ figure).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AHO_DIR = Path("data/embeddings_aho_train012")
SEG = "analysis/aho_analysis/aho_segments.csv"
DATA = "data/uniprot_2022/labeled_sequences.csv"
OUT = Path("analysis/aho_analysis")

names = json.load(open(AHO_DIR / "feature_names.json"))
IDX = {n: i for i, n in enumerate(names)}
CH_PEP_INSIDE = IDX["pep.inside"]
CH_TRAIN = IDX["uniprot_2022.pep.inside"]
CH_EXT = [IDX[k] for k in names
          if k.endswith(".pep.inside") and k.split(".")[0] in
          ("apd6_natural", "dbamp_3", "dramp_general", "dramp_natural")]
ID2STEM = json.load(open(AHO_DIR / "protein_id_to_filename_stem.json"))

_tensor_cache: dict[str, np.ndarray] = {}


def aho_tensor(pid):
    if pid not in _tensor_cache:
        stem = ID2STEM.get(pid)
        if stem is None or not (AHO_DIR / f"{stem}.pt").exists():
            _tensor_cache[pid] = None
        else:
            t = torch.load(AHO_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
            _tensor_cache[pid] = np.asarray(t)
    return _tensor_cache[pid]


def hit_for_span(pid, start, end):
    """Return (aho_hit, source) for 1-based inclusive span [start,end]."""
    t = aho_tensor(pid)
    if t is None:
        return False, "no_embedding"
    span = t[start - 1:end]
    if span.shape[0] == 0:
        return False, "none"
    hit = float(span[:, CH_PEP_INSIDE].max()) > 0
    if not hit:
        return False, "none"
    train_hit = float(span[:, CH_TRAIN].max()) > 0
    ext_hit = any(float(span[:, c].max()) > 0 for c in CH_EXT)
    if train_hit and ext_hit:
        src = "train+external"
    elif train_hit:
        src = "train_only"
    else:
        src = "external_only"
    return True, src


def main():
    seg = pd.read_csv(SEG)
    df = pd.read_csv(DATA, index_col=0)
    pid_seq = dict(zip(df["protein_id"], df["sequence"]))

    # locate each unique (protein_id, seq) span once, compute dict hit
    uniq = seg[["protein_id", "seq"]].drop_duplicates()
    hit_map = {}
    for _, r in uniq.iterrows():
        pid, seq = r["protein_id"], r["seq"]
        prot = pid_seq.get(pid, "")
        pos = prot.find(seq)
        if pos < 0:
            hit_map[(pid, seq)] = (False, "seq_not_found")
            continue
        hit_map[(pid, seq)] = hit_for_span(pid, pos + 1, pos + len(seq))

    seg["aho_hit"] = seg.apply(lambda r: hit_map[(r["protein_id"], r["seq"])][0], axis=1)
    seg["hit_source"] = seg.apply(lambda r: hit_map[(r["protein_id"], r["seq"])][1], axis=1)
    seg.to_csv(OUT / "aho_segments_with_hits.csv", index=False)

    aho_models = [m for m in seg["model"].unique() if m != "baseline"]
    write_summary(seg, aho_models)
    make_figure(seg, aho_models)
    print(f"Wrote {OUT}/dictionary_hit_summary.md")


def _rec(d):
    return d["matched"].mean() if len(d) else float("nan")


def write_summary(seg, aho_models):
    base = seg[seg["model"] == "baseline"]
    L = ["# AHO refinement: stratify by whether the DICTIONARY actually fires", ""]
    L.append("Per true TEST peptide, `aho_hit` = the precomputed `pep.inside` AHO feature "
             "is nonzero somewhere in the peptide span (some dictionary peptide overlaps it). "
             "`hit_source` distinguishes a train(uniprot) hit from an external-AMP-DB-only hit. "
             "Recall uplift = AHO models (mean) − baseline, within each bucket.")
    L.append("")
    # coverage
    cov = base["aho_hit"].mean()
    src = base[base["aho_hit"]]["hit_source"].value_counts().to_dict()
    L.append(f"**Dictionary coverage of true test peptides:** {cov:.1%} have a hit. "
             f"Hit-source breakdown: {src}")
    L.append("")
    L.append("| bucket | n | baseline recall | AHO recall (mean) | uplift |")
    L.append("|---|---:|---:|---:|---:|")
    for label, mask in [
        ("dictionary HIT", base["aho_hit"]),
        ("  ↳ train hit", base["hit_source"].isin(["train_only", "train+external"])),
        ("  ↳ external-only hit", base["hit_source"] == "external_only"),
        ("NO hit", ~base["aho_hit"]),
    ]:
        pids = set(zip(base[mask]["protein_id"], base[mask]["seq"]))
        n = len(base[mask])
        rb = _rec(base[mask])
        recs = []
        for m in aho_models:
            dm = seg[(seg["model"] == m)]
            dm = dm[dm.apply(lambda r: (r["protein_id"], r["seq"]) in pids, axis=1)]
            recs.append(_rec(dm))
        ra = float(np.nanmean(recs)) if recs else float("nan")
        L.append(f"| {label} | {n} | {rb:.3f} | {ra:.3f} | {ra-rb:+.3f} |")
    L.append("")
    L.append("**Reading:** the no-hit bucket isolates peptides the dictionary cannot help "
             "(no signal) — uplift there should be ~0/negative (AHO channel = noise). The "
             "hit buckets show whether AHO actually helps where it fires, and whether an "
             "external-DB hit (peptide novel to train but known to an AMP DB) is exploited "
             "as well as a train hit.")
    (OUT / "dictionary_hit_summary.md").write_text("\n".join(L))


def make_figure(seg, aho_models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = seg[seg["model"] == "baseline"]
    buckets = [("dict HIT\n(train)", base["hit_source"].isin(["train_only", "train+external"])),
               ("dict HIT\n(external-only)", base["hit_source"] == "external_only"),
               ("NO hit", ~base["aho_hit"])]
    labels, bvals, avals, ns = [], [], [], []
    for lab, mask in buckets:
        pids = set(zip(base[mask]["protein_id"], base[mask]["seq"]))
        labels.append(lab); ns.append(len(base[mask])); bvals.append(_rec(base[mask]))
        recs = []
        for m in aho_models:
            dm = seg[seg["model"] == m]
            dm = dm[dm.apply(lambda r: (r["protein_id"], r["seq"]) in pids, axis=1)]
            recs.append(_rec(dm))
        avals.append(float(np.nanmean(recs)))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2, bvals, w, label="baseline (no AHO)", color="#888888")
    ax.bar(x + w/2, avals, w, label="AHO (mean)", color="#3b78c2")
    for i, (b, a) in enumerate(zip(bvals, avals)):
        ax.text(i + w/2, a + .01, f"{a-b:+.3f}", ha="center", fontsize=9, fontweight="bold",
                color=("green" if a >= b else "red"))
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=8)
    ax.set_ylabel("peptide recall (±3)"); ax.set_ylim(0, 1)
    ax.set_title("AHO uplift by actual dictionary hit (and hit source)")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig("texs/error_analysis/figures/aho_uplift_by_dict_hit.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
