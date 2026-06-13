# uniprot_2026 — embedding backfill for the clean-protocol split

The 7-fold motif-balanced split is frozen in `graphpart_assignments.csv` (roles in
`split_roles.json`: train={0,3,4,6}, dev={1}, model_select={2}, test={5}). To train the
ESM-C 6B ⊕ 3Di winner family on it, we need embeddings for every assigned protein. 87% are
reused from `data/uniprot_2022/embeddings/` (keyed by md5(sequence)); only the novel ones
need generating.

## Step 1 — ESM-C 6B (Forge API, rate-limited, ~1055 novel seqs, ~1.5–2 days)
`esmc6b_backfill_novel.fasta` holds the 1055 sequences with no existing ESM-C 6B embedding.
The generator skips files that already exist, so it is resumable — just re-run daily until
the credit limit no longer trips:
```bash
FORGE_TOKEN=<your token> PYTHONPATH=. env/bin/python \
  src/utils/embedding_generators/make_embeddings_esmc_6b.py \
  data/uniprot_2026/esmc6b_backfill_novel.fasta \
  data/uniprot_2022/embeddings/embeddings_esmc6b
```
Output: one `<md5(seq)>.pt` (per-residue, 2560-d) per sequence into the SHARED 2022 dir
(md5 keying makes it dataset-agnostic). Hits "daily credit limit" → stops; resume next day.

## Step 2 — 3Di for the novel seqs (ProstT5, LOCAL, free)
```bash
PYTHONPATH=. env/bin/python src/utils/embedding_generators/make_embeddings_prostt5.py \
  data/uniprot_2026/esmc6b_backfill_novel.fasta <prostt5 3Di out dir>
```

## Step 3 — build the ESM-C 6B ⊕ 3Di concat (2580-d) for the full 2026 assigned set
Concatenate ESM-C 6B (2560) ⊕ 3Di (20) into the embeddings dir the winner config reads.
See `make_embeddings_esm2_3di_concat.py` for the concat pattern; target a 2026 esmc6b_3di
dir. Proteins lacking a 3Di structure are dropped from the 3Di-concat runs (same caveat as
2022: ~1503-vs-1533 there).

## Then
Retrain a PRUNED candidate set (winner family + key baselines, not all 55 runs) on train
folds {0,3,4,6}; select epoch + struct_proj HP on dev (fold 1); pick the architecture on
model_select (fold 2); report the final number on the untouched test (fold 5) + bootstrap CI.
