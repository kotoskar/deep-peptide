"""Run ONE experiment: build config from a base + typed overrides, train, fp32 infer.
Used by queue_driver.py. Run from repo root.

  python run_experiment.py --base runs/<x>/config.json --out <name> --set k=v k2=v2
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()

cfg = json.load(open(a.base))
for kv in a.set:
    k, v = kv.split("=", 1)
    try:
        cfg[k] = json.loads(v)      # int/float/bool/json
    except json.JSONDecodeError:
        cfg[k] = v                  # plain string (e.g. a path)
cfg["out_dir"] = f"runs/{a.out}"
os.makedirs(cfg["out_dir"], exist_ok=True)
json.dump(cfg, open(f"{cfg['out_dir']}/config.json", "w"), indent=3)
print(f"[run_experiment] {a.out}: " + ", ".join(a.set), flush=True)

from argparse import Namespace
from src.train_loop_crf import train
try:
    from aim import Run
    run = Run()
except Exception:
    class _D(dict):
        def track(self, *x, **k): pass
    run = _D()

train(Namespace(**cfg), run=run)
print(f"[run_experiment] {a.out}: training done, fp32 infer...", flush=True)
subprocess.run([sys.executable, "infer.py", "--runs_dir", "runs", "--only", a.out, "--device", "0"],
               env={**os.environ, "PYTHONPATH": "."})
print(f"[run_experiment] {a.out}: DONE", flush=True)
