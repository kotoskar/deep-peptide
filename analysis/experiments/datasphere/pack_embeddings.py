"""Pack only the embeddings one split actually needs, into N tars.

N, not one, because DataSphere caps a single job input file at 5 GiB, and
because a dropped upload then costs one part instead of the whole set.
Hardlinks, so staging costs no extra disk.

Defaults describe the 2026 ESM-2 split; --release 2022 switches all four paths.
"""
import argparse, os, pathlib, subprocess
import pandas as pd
from hashlib import md5

ROOT = pathlib.Path("/home/oskar/work/DeepPeptide")
RELEASES = {
    "2026": dict(
        src="data/uniprot_2026/embeddings/emb_esm2",
        data="data/uniprot_2026/labeled_sequences.csv",
        split="data/uniprot_2026/graphpart_assignments_5motif.esm2covered.csv",
        prefix="emb_part",
        count="analysis/experiments/datasphere/emb_count.txt",
    ),
    "esmc6b": dict(
        src="data/uniprot_2022/embeddings/embeddings_esmc6b",
        data="data/uniprot_2026/labeled_sequences.csv",
        split="data/uniprot_2026/graphpart_assignments_5motif.esmc6bcovered.csv",
        prefix="embc_part",
        count="analysis/experiments/datasphere/emb_count_esmc6b.txt",
    ),
    "2022": dict(
        src="data/uniprot_2022/embeddings/embeddings_esm2",
        data="data/uniprot_2022/labeled_sequences.csv",
        split="data/uniprot_2022/graphpart_assignments.csv",
        prefix="emb22_part",
        count="analysis/experiments/datasphere/emb_count_2022.txt",
    ),
}

ap = argparse.ArgumentParser()
ap.add_argument("--release", choices=sorted(RELEASES), default="2026")
ap.add_argument("--parts", type=int, default=3)
ap.add_argument("--stage", default="/home/oskar/ds_job/stage")
ap.add_argument("--out", default="/home/oskar/ds_job")
a = ap.parse_args()
r = RELEASES[a.release]

SRC = ROOT / r["src"]
STAGE = pathlib.Path(a.stage) / a.release
OUT = pathlib.Path(a.out)
N_PARTS = a.parts

data = pd.read_csv(ROOT / r["data"], index_col="protein_id")
part = pd.read_csv(ROOT / r["split"], index_col="AC")
d = data.join(part, how="inner")
names = []
for pid, row in d.iterrows():
    h = md5(str(row["sequence"]).encode()).hexdigest() + ".pt"
    if (SRC / h).exists():
        names.append(h)
# Different accessions can carry an identical sequence (461 of them in the 2026
# split, e.g. P69556/P69557), and an embedding file is keyed by md5(sequence),
# so the file count is the number of UNIQUE sequences, not the number of proteins.
names = sorted(set(names))
COUNT = ROOT / r["count"]
COUNT.write_text(f"{len(names)}\n")
print(f"[pack] {a.release}: {len(names)} unique embedding files needed -> {COUNT}", flush=True)

# round-robin so the parts come out the same size
for i in range(N_PARTS):
    (STAGE / f"part{i}").mkdir(parents=True, exist_ok=True)
for j, h in enumerate(names):
    dst = STAGE / f"part{j % N_PARTS}" / h
    if not dst.exists():
        os.link(SRC / h, dst)

for i in range(N_PARTS):
    p = STAGE / f"part{i}"
    n = len(list(p.iterdir()))
    size = sum(f.stat().st_size for f in p.iterdir())
    tar = OUT / f"{r['prefix']}{i}.tar"
    print(f"[pack] part{i}: {n} files, {size/2**30:.2f} GiB -> {tar}", flush=True)
    subprocess.check_call(["tar", "-cf", str(tar), "-C", str(p), "."])
print("[pack] done", flush=True)
