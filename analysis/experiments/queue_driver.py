"""Resumable sequential experiment driver. Runs each queued experiment (train + fp32
infer) one at a time, waiting for the GPU to be free first (so it won't collide with an
externally-launched training like seq512). Reboot-safe: skips any experiment that
already has runs/<name>/test_metrics_infer.json. Writes a status file each step.

  python queue_driver.py <queue.jsonl> <status.json>
Run from repo root with PYTHONPATH=.
"""
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(".").resolve()
QUEUE = Path(sys.argv[1])
STATUS = Path(sys.argv[2]) if len(sys.argv) > 2 else (QUEUE.parent / "queue_status.json")
HERE = Path(__file__).parent
# external trainings we must wait out (launched outside the driver)
EXTERNAL_PATTERNS = "launch_3di_seq512.py|launch_boundary|launch_gated3di_boundary.py"


def gpu_occupied():
    out = subprocess.run(["pgrep", "-f", EXTERNAL_PATTERNS], capture_output=True, text=True)
    return bool(out.stdout.strip())


def write_status(**kw):
    kw["ts"] = datetime.now().isoformat(timespec="seconds")
    json.dump(kw, open(STATUS, "w"), indent=2)


def main():
    items = [json.loads(l) for l in open(QUEUE) if l.strip()]
    for i, it in enumerate(items):
        name = it["name"]
        done_marker = ROOT / "runs" / name / "test_metrics_infer.json"
        if done_marker.exists():
            print(f"[skip done] {name}", flush=True)
            write_status(stage="skip", current=name, idx=i, total=len(items))
            continue
        while gpu_occupied():
            write_status(stage="waiting_gpu", current=name, idx=i, total=len(items))
            time.sleep(60)
        write_status(stage="running", current=name, idx=i, total=len(items))
        cmd = [sys.executable, str(HERE / "run_experiment.py"),
               "--base", it["base"], "--out", name, "--set", *it.get("set", [])]
        print(f"[launch {i+1}/{len(items)}] {name}", flush=True)
        rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "."}).returncode
        ok = done_marker.exists()
        write_status(stage="finished" if ok else "FAILED", current=name, idx=i,
                     total=len(items), returncode=rc)
        print(f"[{'done' if ok else 'FAILED'}] {name} rc={rc}", flush=True)
    write_status(stage="QUEUE_COMPLETE", total=len(items))
    print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
