#!/usr/bin/env python3
"""Peptide-level similarity of held-out (valid/test) peptides to TRAIN peptides.

Motivation: the train/val/test split is homology-separated at the WHOLE-PROTEIN
level (GraphPart `needle --threshold 0.3`). That does NOT guarantee the peptide
*segments* themselves are novel — conserved peptide motifs can recur across
non-homologous proteins. This script measures, for every unique valid/test
peptide, its maximum identity to any TRAIN peptide, so we can bucket held-out
peptides into "similar-to-train" vs "novel" (supervisor's 70% threshold) and
later correlate that with model recall (the data-ceiling argument) and the AHO
analysis.

Method:
  * Extract peptide and propeptide segments (merged coordinates), per split.
  * Dedupe to UNIQUE sequences per (split, type), keeping occurrence metadata.
  * Align each held-out unique peptide vs all TRAIN unique peptides of the SAME
    type with EMBOSS `needleall` (global Needleman-Wunsch — same aligner family
    GraphPart used). Identity = matches / alignment_length (needle default,
    GraphPart-consistent). We ALSO record coverage = matches / min(len_q,len_db)
    to expose the "short peptide fully contained in a longer one" case, which
    the alignment-length identity under-counts.
  * For each held-out peptide keep the best (max-identity) train hit.

Outputs (analysis/peptide_similarity/):
  * peptide_similarity.csv  — one row per unique held-out peptide:
      seq,type,split,length,n_occurrences,organisms,
      max_identity_to_train, coverage_at_best, best_train_seq,
      is_similar_70 (identity>=0.70)
  * summary.md / figures.

Usage: env/bin/python analysis/peptide_similarity.py [--device unused]
Requires EMBOSS needleall on PATH.
"""
from __future__ import annotations
import subprocess, sys, tempfile, os
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.crf_label_utils import parse_coordinate_string

DATA = "data/uniprot_2022/labeled_sequences.csv"
GP = "data/uniprot_2022/graphpart_assignments.csv"
SPLIT = {0: "train", 1: "train", 2: "train", 3: "valid", 4: "test"}
OUT = Path("analysis/peptide_similarity"); OUT.mkdir(parents=True, exist_ok=True)
MIN_LEN, MAX_LEN = 5, 100
SIM_THRESHOLD = 0.70


def species(o):
    return o.split("(")[0].strip() if isinstance(o, str) else "unknown"


def extract():
    """Return uniq[(split,type)] -> {seq: {'count':int,'orgs':set,'pids':set}}."""
    df = pd.read_csv(DATA, index_col=0)
    gp = pd.read_csv(GP, index_col="AC")
    uniq = defaultdict(lambda: defaultdict(lambda: {"count": 0, "orgs": set(), "pids": set()}))
    for _, row in df.iterrows():
        ac = row["protein_id"]
        if ac not in gp.index:
            continue
        sp = SPLIT.get(int(gp.loc[ac, "cluster"]))
        if sp is None:
            continue
        seq = row["sequence"]; org = species(row.get("organism"))
        for col, typ in [("coordinates", "pep"), ("propeptide_coordinates", "propep")]:
            if pd.isna(row[col]):
                continue
            for st, en in parse_coordinate_string(str(row[col]), merge_overlaps=True):
                pep = seq[st - 1:en]
                if MIN_LEN <= len(pep) <= MAX_LEN:
                    rec = uniq[(sp, typ)][pep]
                    rec["count"] += 1; rec["orgs"].add(org); rec["pids"].add(ac)
    return uniq


def write_fasta(seqs, path):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">q{i}\n{s}\n")
    return {f"q{i}": s for i, s in enumerate(seqs)}


def needleall_max_identity(query_seqs, db_seqs, workdir):
    """Stream needleall query-vs-db; return {query_seq: (max_id, coverage, best_db_seq)}."""
    qmap = write_fasta(query_seqs, workdir / "q.fa")
    dmap = write_fasta(db_seqs, workdir / "db.fa")
    qlen = {k: len(v) for k, v in qmap.items()}
    dlen = {k: len(v) for k, v in dmap.items()}
    best = {q: (-1.0, 0.0, None) for q in qmap}  # qid -> (identity, coverage, dbid)

    proc = subprocess.Popen(
        ["needleall", "-auto", "-asequence", str(workdir / "q.fa"),
         "-bsequence", str(workdir / "db.fa"), "-gapopen", "10", "-gapextend", "0.5",
         "-aformat3", "pair", "-errfile", str(workdir / "err.txt"),
         "-outfile", "stdout"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    cur_q = cur_d = None
    for line in proc.stdout:
        if line.startswith("# 1:"):
            cur_q = line.split(":", 1)[1].strip()
        elif line.startswith("# 2:"):
            cur_d = line.split(":", 1)[1].strip()
        elif line.startswith("# Identity:"):
            frac = line.split(":", 1)[1].strip().split("(")[0].strip()  # "X/Y"
            x, y = frac.split("/")
            matches, aln_len = int(x), int(y)
            ident = matches / aln_len if aln_len else 0.0
            if cur_q in best and ident > best[cur_q][0]:
                cov = matches / min(qlen[cur_q], dlen[cur_d]) if cur_d in dlen else 0.0
                best[cur_q] = (ident, cov, cur_d)
    proc.wait()
    return {qmap[q]: (v[0], v[1], dmap.get(v[2])) for q, v in best.items()}


def main():
    uniq = extract()
    for k in sorted(uniq):
        print(f"  {k[0]:6s} {k[1]:7s} unique={len(uniq[k])}")

    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for typ in ["pep", "propep"]:
            train_seqs = list(uniq[("train", typ)].keys())
            for split in ["valid", "test"]:
                q_seqs = list(uniq[(split, typ)].keys())
                if not q_seqs or not train_seqs:
                    continue
                print(f"aligning {split}/{typ}: {len(q_seqs)} x {len(train_seqs)} ...", flush=True)
                wd = td / f"{split}_{typ}"; wd.mkdir()
                res = needleall_max_identity(q_seqs, train_seqs, wd)
                for seq, (ident, cov, best_db) in res.items():
                    rec = uniq[(split, typ)][seq]
                    rows.append({
                        "seq": seq, "type": typ, "split": split, "length": len(seq),
                        "n_occurrences": rec["count"],
                        "organisms": ";".join(sorted(rec["orgs"])),
                        "max_identity_to_train": round(ident, 4),
                        "coverage_at_best": round(cov, 4),
                        "best_train_seq": best_db,
                        "is_similar_70": ident >= SIM_THRESHOLD,
                    })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "peptide_similarity.csv", index=False)
    write_summary(out_df)
    print(f"\nWrote {OUT}/peptide_similarity.csv ({len(out_df)} unique held-out peptides)")


def write_summary(df):
    lines = ["# Peptide-level similarity of held-out peptides to TRAIN", ""]
    lines.append("Per unique valid/test peptide: max identity (needle, "
                 "matches/alignment_length) to any train peptide of the same type. "
                 f"`is_similar_70` = identity ≥ {SIM_THRESHOLD:.0%}. `coverage_at_best` "
                 "= matches/min(len) at the best hit (catches short-in-long containment).")
    lines.append("")
    lines.append("> The split is homology-separated at the WHOLE-PROTEIN level "
                 "(GraphPart needle, 30%). This measures whether the peptide SEGMENTS "
                 "are also novel.")
    lines.append("")
    for split in ["valid", "test"]:
        for typ in ["pep", "propep"]:
            sub = df[(df["split"] == split) & (df["type"] == typ)]
            if not len(sub):
                continue
            sim = (sub["max_identity_to_train"] >= SIM_THRESHOLD).mean()
            cov_sim = (sub["coverage_at_best"] >= SIM_THRESHOLD).mean()
            med = sub["max_identity_to_train"].median()
            lines.append(f"- **{split}/{typ}** (n={len(sub)} unique): median max-identity "
                         f"{med:.2f}; **{sim:.0%} ≥70% identity** to a train peptide; "
                         f"{cov_sim:.0%} ≥70% coverage (containment).")
    lines.append("")
    # identity histogram (text)
    lines.append("## Max-identity-to-train distribution (test, both types)")
    lines.append("")
    lines.append("| identity bin | pep | propep |")
    lines.append("|---|---:|---:|")
    bins = [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    labels = ["<0.30", "0.30–0.50", "0.50–0.70", "0.70–0.90", "0.90–1.00"]
    for (lo, hi), lab in zip(bins, labels):
        t = df[df["split"] == "test"]
        npep = ((t[t["type"] == "pep"]["max_identity_to_train"] >= lo) &
                (t[t["type"] == "pep"]["max_identity_to_train"] < hi)).sum()
        npro = ((t[t["type"] == "propep"]["max_identity_to_train"] >= lo) &
                (t[t["type"] == "propep"]["max_identity_to_train"] < hi)).sum()
        lines.append(f"| {lab} | {npep} | {npro} |")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
