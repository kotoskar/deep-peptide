#!/usr/bin/env python3
"""Segment F1 against segment length, under the nested CV.

This is the link the fold-exchangeability argument needs. The folds differ in
segment-length composition, and quality depends strongly on segment length, so a
fold's length profile sets how hard it is before any model touches it.

    env/bin/python analysis/metrics/src/generators/fig_bylength_cv.py
"""
from __future__ import annotations
import argparse, json, pathlib, sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paperstyle import apply, SERIES_STYLE, textwidth  # noqa: E402

SHOW = [
    ("5cv_baseline_esm2_fp32", "ESM-2, base", "base_esm2"),
    ("5cv_esm2_full",          "ESM-2 + both", "full"),
    ("5cv_esmc6b_plain",       "ESM-C 6B, base", "base_esmc"),
    ("5cv_esmc6b_full",        "ESM-C 6B + both", "full_esmc"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="analysis/metrics/bylength_cv.json")
    ap.add_argument("--outdir", default="texs/ai4dd/figures")
    args = ap.parse_args()

    d = json.load(open(args.json))
    bins = d["kept_bins"]
    x = list(range(len(bins)))

    apply()
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(textwidth(1.0), 2.9), height_ratios=[2.6, 1.0],
        sharex=True, layout="constrained")

    for run, label, key in SHOW:
        m = d["models"].get(run)
        if not m:
            print(f"[skip] {run}")
            continue
        colour, marker, dash = SERIES_STYLE[key]
        ys = [m["metrics"][f"{b}_f1"]["mean"] for b in bins]
        ax.plot(x, ys, color=colour, marker=marker, linestyle=dash,
                markersize=3.4, linewidth=1.3, label=label)

    base = d["models"][SHOW[0][0]]["metrics"]
    counts = [base[f"{b}_n_true"]["mean"] for b in bins]
    axn.bar(x, counts, color="#9AA5A6", alpha=0.55, width=0.66, linewidth=0)

    ax.set_ylabel("segment F1 at $\\pm3$")
    axn.set_ylabel("true segments\nper cell")
    axn.set_xticks(x)
    axn.set_xticklabels([b.replace("-", "--") for b in bins])
    axn.set_xlabel("segment length (residues)")
    for a in (ax, axn):
        a.margins(x=0.04)
    fig.legend(loc="outside lower center", ncols=2, handlelength=2.4, columnspacing=1.6)

    out = pathlib.Path(args.outdir) / "fig_bylength_cv.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")

    ys = [base[f"{b}_f1"]["mean"] for b in bins]
    lo, hi = min(ys), max(ys)
    print(f"base F1 ranges {lo:.3f} ({bins[ys.index(lo)]}) to {hi:.3f} ({bins[ys.index(hi)]}), "
          f"a spread of {hi - lo:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
