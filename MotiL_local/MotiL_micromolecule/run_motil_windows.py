"""
Windows launcher for the micromolecule part of MotiL.
Run from this folder, for example:
  python run_motil_windows.py finetune --task bbbp --device cpu
  python run_motil_windows.py finetune --task all --device cuda --gpu 0
  python run_motil_windows.py pretrain --device cpu
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "dumped" / "pre-train" / "1-model" / "original_CMPN_0707_0800_12000th_epoch.pkl"

TASKS = {
    "bbbp": [
        "train.py", "--data_path", "./data/bbbp.csv", "--exp_id", "bbbp",
        "--dataset_type", "classification", "--epochs", "10", "--num_runs", "3",
        "--batch_size", "16", "--seed", "2024", "--init_lr", "1e-5",
        "--final_lr", "5e-6", "--wd", "5e-5", "--warmup_epochs", "0.0",
        "--split_type", "scaffold_balanced", "--exp_name", "finetune",
        "--split_sizes", "0.8", "0.1", "0.1", "--step", "finetune",
        "--checkpoint_path", str(CHECKPOINT),
    ],
    "clintox": [
        "train.py", "--data_path", "./data/clintox.csv", "--exp_id", "clintox",
        "--dataset_type", "classification", "--epochs", "10", "--num_runs", "3",
        "--batch_size", "16", "--seed", "2024", "--init_lr", "1e-5",
        "--final_lr", "5e-6", "--wd", "5e-5", "--warmup_epochs", "0.0",
        "--split_type", "scaffold_balanced", "--exp_name", "finetune",
        "--split_sizes", "0.8", "0.1", "0.1", "--step", "finetune",
        "--checkpoint_path", str(CHECKPOINT),
    ],
    "bace": [
        "train.py", "--data_path", "./data/bace.csv", "--exp_id", "bace",
        "--dataset_type", "classification", "--epochs", "20", "--num_runs", "3",
        "--batch_size", "16", "--seed", "2024", "--init_lr", "1e-3",
        "--final_lr", "5e-4", "--wd", "1e-5", "--warmup_epochs", "0.0",
        "--split_type", "scaffold_balanced", "--exp_name", "finetune",
        "--split_sizes", "0.8", "0.1", "0.1", "--step", "finetune",
        "--checkpoint_path", str(CHECKPOINT),
    ],
    "esol": [
        "train.py", "--data_path", "./data/esol.csv", "--dataset_type", "regression",
        "--epochs", "100", "--num_runs", "3", "--batch_size", "64",
        "--seed", "2024", "--init_lr", "1e-3", "--final_lr", "1e-6",
        "--wd", "1e-5", "--warmup_epochs", "0.0", "--split_type", "scaffold_balanced",
        "--exp_name", "scaffold_balanced", "--exp_id", "esol", "--metric", "rmse",
        "--checkpoint_path", str(CHECKPOINT),
    ],
}


def with_device(cmd: list[str], device: str, gpu: int | None) -> list[str]:
    if device == "cuda":
        return [*cmd, "--gpu", str(0 if gpu is None else gpu)]
    return [*cmd, "--no_cuda"]


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MotiL micromolecule tasks on Windows.")
    sub = parser.add_subparsers(dest="mode", required=True)

    finetune = sub.add_parser("finetune", help="Fine-tune on BBBP, ClinTox, BACE, ESOL, or all.")
    finetune.add_argument("--task", choices=[*TASKS.keys(), "all"], default="bbbp")
    finetune.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    finetune.add_argument("--gpu", type=int, default=None)

    pretrain = sub.add_parser("pretrain", help="Run the original pretraining command on ZINC15 250K.")
    pretrain.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    pretrain.add_argument("--gpu", type=int, default=None)

    for subparser in (finetune, pretrain):
        subparser.add_argument("--epochs", type=int)
        subparser.add_argument("--batch-size", type=int)
        subparser.add_argument("--max-data-size", type=int)
        subparser.add_argument("--exp-id")
        subparser.add_argument("--smoke", action="store_true", help="Short execution check; not a benchmark")
    finetune.add_argument("--num-runs", type=int, default=3)
    finetune.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    pretrain.add_argument("--data", type=Path, default=ROOT / 'data/zinc15_250K.csv')

    args = parser.parse_args()

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu or repair the PyTorch environment.')
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.batch_size is not None and args.batch_size < 2:
        parser.error('--batch-size must be at least 2')
    if args.max_data_size is not None and args.max_data_size < 1:
        parser.error('--max-data-size must be positive')
    if args.mode == 'finetune' and args.num_runs < 1:
        parser.error('--num-runs must be positive')
    run_id = args.exp_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def override(cmd, key, value):
        if key in cmd:
            index = cmd.index(key)
            cmd[index + 1] = str(value)
        else:
            cmd.extend([key, str(value)])

    def customize(cmd):
        cmd = list(cmd)
        override(cmd, '--exp_id', run_id)
        override(cmd, '--exp_name', 'local_' + args.mode)
        if args.epochs is not None or args.smoke:
            override(cmd, '--epochs', args.epochs or 1)
        if args.batch_size is not None or args.smoke:
            override(cmd, '--batch_size', args.batch_size or 8)
        if args.max_data_size is not None:
            override(cmd, '--max_data_size', args.max_data_size)
        return cmd

    if args.mode == "pretrain":
        cmd = ['pretrain.py', '--data_path', str(args.data.resolve()), '--step', 'pretrain', '--epochs', '50', '--batch_size', '32']
        cmd = customize(cmd)
        if args.smoke and args.max_data_size is None:
            override(cmd, '--max_data_size', 32)
        if args.smoke:
            size = args.max_data_size or 32
            batch = args.batch_size or 8
            if size // batch < 2:
                parser.error('Pretraining smoke check requires at least two batches for molecule and motif losses.')
        run(with_device(cmd, args.device, args.gpu))
        return

    tasks = TASKS.keys() if args.task == "all" else [args.task]
    for task in tasks:
        cmd = customize(TASKS[task])
        override(cmd, '--exp_id', run_id + '_' + task)
        override(cmd, '--num_runs', 1 if args.smoke else args.num_runs)
        override(cmd, '--checkpoint_path', args.checkpoint.resolve())
        if not args.checkpoint.is_file():
            parser.error(f'Checkpoint is missing: {args.checkpoint}')
        run(with_device(cmd, args.device, args.gpu))


if __name__ == "__main__":
    main()
