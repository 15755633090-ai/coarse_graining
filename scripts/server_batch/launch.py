"""Launch the six portable readout jobs concurrently across the supplied GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scripts.readout_ablation.models import TRAIN_VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=8)
    options = parser.parse_args()
    if not options.gpus or len(set(options.gpus)) != len(options.gpus):
        parser.error("Specify distinct GPU indices")
    if options.micro_batch_size < 1 or options.cpu_threads < 1:
        parser.error("micro-batch-size and cpu-threads must be positive")

    jobs = [(variant, seed) for variant in TRAIN_VARIANTS for seed in range(3)]
    processes = []
    for index, (variant, seed) in enumerate(jobs):
        gpu = options.gpus[index % len(options.gpus)]
        job_dir = options.output_dir / f"{variant}_seed_{seed}"
        log_path = job_dir / "console.log"
        job_dir.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        command = [
            sys.executable, "-u", "-m", "scripts.server_batch.run_job",
            "--bundle", str(options.bundle.resolve()),
            "--output-dir", str(job_dir.resolve()),
            "--variant", variant, "--seed", str(seed),
            "--device", "cuda:0",
            "--micro-batch-size", str(options.micro_batch_size),
            "--cpu-threads", str(options.cpu_threads),
        ]
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        processes.append((variant, seed, gpu, process, handle, log_path))
        print(f"Started {variant} seed={seed} on physical GPU {gpu}: {log_path}", flush=True)

    failed = []
    for variant, seed, gpu, process, handle, log_path in processes:
        code = process.wait()
        handle.close()
        if code:
            failed.append(f"{variant}/seed_{seed} on GPU {gpu} (exit {code}; see {log_path})")
    if failed:
        raise SystemExit("Failed jobs:\n" + "\n".join(failed))
    print("All six validation-only jobs completed.", flush=True)


if __name__ == "__main__":
    main()
