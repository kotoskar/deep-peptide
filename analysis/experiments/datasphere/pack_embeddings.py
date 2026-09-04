"""Pack only the 8,897 embeddings the 2026 ESM-2 split actually needs, into 3 tars.

Three, not one, because DataSphere caps a single job input file at 5 GiB, and
because a dropped upload then costs one part instead of the whole 9.2 GiB.
Hardlinks, so staging costs no extra disk.
"""
import os, pathlib, subprocess, sys
import pandas as pd
from hashlib import md5

ROOT = pathlib.Path("/home/oskar/work/DeepPeptide")
SRC = ROOT / "data/uniprot_2026/embeddings/emb_esm2"
STAGE = pathlib.Path("/home/oskar/ds_job/stage")
OUT = pathlib.Path("/home/oskar/ds_job")
N_PARTS = 3

data = pd.read_csv(ROOT / "data/uniprot_2026/labeled_sequences.csv", index_col="protein_id")
part = pd.read_csv(ROOT / "data/uniprot_2026/graphpart_assignments_5motif.esm2covered.csv", index_col="AC")
d = data.join(part, how="inner")
names = []
for pid, row in d.iterrows():
    h = md5(str(row["sequence"]).encode()).hexdigest() + ".pt"
    if (SRC / h).exists():
        names.append(h)
names.sort()
print(f"[pack] {len(names)} embedding files needed", flush=True)

# round-robin so the parts come out the same size
for i in range(N_PARTS):
    (STAGE / f"emb_part{i}").mkdir(parents=True, exist_ok=True)
for j, h in enumerate(names):
    dst = STAGE / f"emb_part{j % N_PARTS}" / h
    if not dst.exists():
        os.link(SRC / h, dst)

for i in range(N_PARTS):
    p = STAGE / f"emb_part{i}"
    n = len(list(p.iterdir()))
    size = sum(f.stat().st_size for f in p.iterdir())
    tar = OUT / f"emb_part{i}.tar"
    print(f"[pack] part{i}: {n} files, {size/2**30:.2f} GiB -> {tar}", flush=True)
    subprocess.check_call(["tar", "-cf", str(tar), "-C", str(p), "."])
print("[pack] done", flush=True)
