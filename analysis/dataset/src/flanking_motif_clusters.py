#!/usr/bin/env python3
"""Reproduce the DeepPeptide flanking-motif clustering used to BALANCE the GraphPart folds.

Paper (Teufel et al., Methods): "We extracted the two AAs before the N and after the C
terminus of each peptide. To capture biochemical characteristics, we encoded each AA using
ESM-2 and concatenated the resulting four embedding vectors. On the concatenated vectors, we
performed k-means clustering ... k=50 ... captures meaningful motifs while still amenable to
partitioning." The resulting per-protein cluster label is fed to GraphPart (`--labels-name`)
to balance the folds so heterogeneous peptide classes don't drift between partitions.

Our 2026 build SKIPPED this (plain `graphpart needle`, no label). This script adds it.

Reconstruction choices (paper underspecifies; documented in the clean-eval-protocol memo):
  * Segments = peptide AND propeptide cleavage sites (paper says "of the peptides", but 55%
    of proteins are propeptide-only; without propep flanks they'd be unlabeled).
  * Flank window for a 1-based segment [s, e]: positions {s-2, s-1, e+1, e+2} (the 4 residues
    OUTSIDE the segment, i.e. the protease-recognition context). Missing positions at protein
    termini -> zero vector. Concatenation order: [N-2, N-1, C+1, C+2] -> 4*1280 = 5120-d.
  * ESM-2 = our cached per-residue embeddings emb_esm2 (keyed md5(sequence)); for fold
    BALANCING the exact ESM-2 variant is immaterial.
  * Per-protein label (GraphPart needs one/seq) = most frequent segment cluster (tie -> min id).

Out: a motif-labelled FASTA (`>id|label=<cluster>`) for GraphPart + a record CSV.

Run from repo root: env/bin/python analysis/dataset/src/flanking_motif_clusters.py
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
DATA = ROOT / "data" / "uniprot_2026"
EMB = DATA / "embeddings" / "emb_esm2"
EMB_DIM = 1280
K = 50
SEED = 42


def parse_coords(coord_string):
    if pd.isna(coord_string):
        return []
    s = str(coord_string).strip()
    if s == "" or s.lower() == "nan":
        return []
    out = []
    for part in s.split(","):
        part = part.strip().lstrip("(").rstrip(")")
        if not part:
            continue
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def flank_vector(emb: torch.Tensor, s: int, e: int) -> np.ndarray:
    """emb: (L,1280) per-residue. 1-based segment [s,e]. Returns 5120-d [N-2,N-1,C+1,C+2]."""
    L = emb.shape[0]
    zero = np.zeros(EMB_DIM, dtype=np.float32)
    pos = [s - 2, s - 1, e + 1, e + 2]  # 1-based positions outside the segment
    vecs = []
    for p in pos:
        idx = p - 1  # to 0-based
        vecs.append(emb[idx].numpy().astype(np.float32) if 0 <= idx < L else zero)
    return np.concatenate(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out_fasta", default=str(DATA / "protein_sequences.motif.fasta"))
    ap.add_argument("--out_csv", default=str(DATA / "processed_data" / "flanking_motif_clusters.csv"))
    args = ap.parse_args()

    df = pd.read_csv(DATA / "labeled_sequences.csv")
    print(f"proteins: {len(df)}")

    seg_vecs, seg_owner = [], []   # parallel: 5120-d vector, protein_id
    missing_emb, n_seg = 0, 0
    emb_cache_misses = set()
    for row in df.itertuples(index=False):
        seq = str(row.sequence)
        h = md5(seq.encode()).hexdigest()
        f = EMB / f"{h}.pt"
        if not f.exists():
            emb_cache_misses.add(row.protein_id)
            continue
        emb = torch.load(f, weights_only=False)
        if emb.shape[0] != len(seq):
            # embedding length mismatch (e.g. truncated) -> skip protein
            missing_emb += 1
            continue
        segs = parse_coords(row.coordinates) + parse_coords(row.propeptide_coordinates)
        for (s, e) in segs:
            seg_vecs.append(flank_vector(emb, s, e))
            seg_owner.append(row.protein_id)
            n_seg += 1

    print(f"segments with embedding: {n_seg}; proteins missing emb file: {len(emb_cache_misses)}; "
          f"length-mismatch skips: {missing_emb}")
    X = np.vstack(seg_vecs)
    print(f"feature matrix: {X.shape} (expect 4*{EMB_DIM}={4*EMB_DIM})")

    km = KMeans(n_clusters=args.k, random_state=SEED, n_init=10)
    seg_cluster = km.fit_predict(X)
    print(f"k-means k={args.k} done; cluster sizes (min/median/max): "
          f"{np.min(np.bincount(seg_cluster))}/{int(np.median(np.bincount(seg_cluster)))}/"
          f"{np.max(np.bincount(seg_cluster))}")

    # per-protein label = most frequent segment cluster (tie -> smallest id)
    by_prot = {}
    for pid, c in zip(seg_owner, seg_cluster):
        by_prot.setdefault(pid, []).append(int(c))
    prot_label = {}
    for pid, cs in by_prot.items():
        cnt = Counter(cs)
        top = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        prot_label[pid] = top

    # proteins with no usable segment/embedding -> assign a dedicated label (k) so GraphPart still places them
    no_seg = [p for p in df.protein_id if p not in prot_label]
    for p in no_seg:
        prot_label[p] = args.k  # sentinel "unlabelled" bucket
    print(f"proteins labelled by motif: {len(prot_label)-len(no_seg)}; "
          f"no-segment/no-emb -> sentinel bucket {args.k}: {len(no_seg)}")

    # record CSV
    rec = pd.DataFrame({"protein_id": list(prot_label), "motif_cluster": list(prot_label.values())})
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(args.out_csv, index=False)

    # motif-labelled FASTA for GraphPart: header '>id|label=<cluster>'
    seqs = df.set_index("protein_id")["sequence"].to_dict()
    with open(args.out_fasta, "w") as fh:
        for pid, lab in prot_label.items():
            s = str(seqs[pid])
            fh.write(f">{pid}|label={lab}\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")
    print(f"wrote {args.out_fasta} and {args.out_csv}")
    print("\nNEXT (cheap, reuses cached NW edges):\n"
          f"  graphpart precomputed --fasta-file {args.out_fasta} \\\n"
          f"    --edge-file {DATA}/processed_data/graphpart_edges_threshold_30.csv \\\n"
          f"    --threshold 0.3 --partitions 7 --no-moving --labels-name label \\\n"
          f"    --out-file {DATA}/graphpart_assignments_7motif.raw.csv")


if __name__ == "__main__":
    main()
