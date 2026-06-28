"""Launch an adapter-isolation ablation by CLONING zeroctrl's exact recipe and
overriding ONLY seq_proj_size + gated_seq_only + out_dir.

zeroctrl = lstmcnncrf_gated3di_boundary on esmc6b_3dizero embeddings, clean
2026 split (train [0,3,4,6], valid [1], test/model-select [2]; fold 5 sealed).
seq_only=True drops the struct/3Di branch -> pure sequence adapter
(LayerNorm+reproject) on top of the boundary head. Same covered split (8979)
as orange esmc6b_boundary, so orange->this isolates the adapter.

Usage: env/bin/python train_adapter_ablation.py <seq_proj> <out_dir> [epochs]
"""
import os, sys, json
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from src.train_loop_crf import train

SEQ_PROJ = int(sys.argv[1])
OUT_DIR  = sys.argv[2]
EPOCHS   = int(sys.argv[3]) if len(sys.argv) > 3 else 100

cfg = json.load(open("runs/2026_esmc6b_3di_zeroctrl/config.json"))
# --- the ablation overrides (everything else = zeroctrl) ---
cfg["seq_proj_size"]  = SEQ_PROJ
cfg["gated_seq_only"] = True
cfg["out_dir"]        = OUT_DIR
cfg["epochs"]         = EPOCHS
cfg["checkpoints_dir"] = os.path.join(OUT_DIR, "checkpoints")
args = SimpleNamespace(**cfg)

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(vars(args), open(os.path.join(OUT_DIR, "config.json"), "w"), indent=3)
with open(os.path.join(OUT_DIR, "SPLIT_ROLES.txt"), "w") as f:
    f.write("2026 clean-protocol split. train=[0,3,4,6] valid=[1] model_select(test)=[2]. "
            "Fold 5 = SEALED. ADAPTER ISOLATION ablation: gated_seq_only=True (no struct/3Di), "
            f"seq_proj_size={SEQ_PROJ}. Clone of esmc6b_3di_zeroctrl recipe.\n")

# aim-free stub: train() only does run['hparams']=... and aim_run.track(...)
class RunStub:
    def __setitem__(self, k, v): pass
    def __getitem__(self, k): return {}
    def track(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: None

print(f"[launch] seq_proj={SEQ_PROJ} epochs={EPOCHS} out={OUT_DIR} "
      f"model={args.model} embeddings={args.embeddings_dir} seq_only={args.gated_seq_only}", flush=True)
train(args, train_partitions=[0, 3, 4, 6], valid_partitions=[1], test_partitions=[2], run=RunStub())
print(f"[done] {OUT_DIR}", flush=True)
