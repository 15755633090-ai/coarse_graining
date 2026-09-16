"""Run a six-job portable readout queue with at most one job per GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.readout_ablation.models import TRAIN_VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    options = parser.parse_args()
    if not options.gpus or len(set(options.gpus)) != len(options.gpus):
        parser.error("Specify distinct GPU indices")
    manifest = json.loads((options.bundle / "manifest.json").read_text(encoding="utf-8"))
    execution = manifest["server_execution"]

    jobs = [(variant, seed) for variant in TRAIN_VARIANTS for seed in manifest["protocol_scope"]["seeds"]]
    pending = iter(jobs)
    active = {}

    def start(gpu: int, variant: str, seed: int) -> None:
        job_dir = options.output_dir / f"{variant}_seed_{seed}"
        log_path = job_dir / "console.log"
        job_dir.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        environment["CUBLAS_WORKSPACE_CONFIG"] = execution["cublas_workspace_config"]
        command = [
            sys.executable, "-u", "-m", "scripts.server_batch.run_job",
            "--bundle", str(options.bundle.resolve()),
            "--output-dir", str(job_dir.resolve()),
            "--variant", variant, "--seed", str(seed),
        ]
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        active[gpu] = (variant, seed, process, handle, log_path)
        print(f"Started {variant} seed={seed} on physical GPU {gpu}: {log_path}", flush=True)

    for gpu in options.gpus:
        try:
            variant, seed = next(pending)
        except StopIteration:
            break
        start(gpu, variant, seed)

    failed = []
    while active:
        for gpu, (variant, seed, process, handle, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[gpu]
            if code:
                failed.append(f"{variant}/seed_{seed} on GPU {gpu} (exit {code}; see {log_path})")
            else:
                print(f"Completed {variant} seed={seed} on physical GPU {gpu}", flush=True)
            try:
                next_variant, next_seed = next(pending)
            except StopIteration:
                continue
            start(gpu, next_variant, next_seed)
        time.sleep(2)
    if failed:
        raise SystemExit("Failed jobs:\n" + "\n".join(failed))
    print("All six validation-only jobs completed.", flush=True)


if __name__ == "__main__":
    main()
