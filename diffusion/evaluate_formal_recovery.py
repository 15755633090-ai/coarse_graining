from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bond_diffusion.config import ModelConfig
from bond_diffusion.data import collate_graphs
from bond_diffusion.evaluation import (
    load_diffusion_checkpoint,
    load_graph_dataset,
    require_non_leaky_checkpoint,
)
from bond_diffusion.formal_evaluation import (
    compute_frequency_modes,
    evaluate_formal_recovery_once,
    save_formal_recovery,
    summarize_formal_runs,
)
from bond_diffusion.model import BondAwareDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal node/bond/valence/substructure recovery benchmark"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--frequency-reference",
        type=Path,
        help=(
            "clean pretraining corpus used to fit frequency modes; required "
            "when the frequency method is selected"
        ),
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--noise", type=float, nargs="+", default=(0.1, 0.2, 0.3))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("pretrained", "random", "frequency"),
        default=("pretrained", "random", "frequency"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/formal_recovery")
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed method/noise/repeat units from the progress file",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    require_non_leaky_checkpoint(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "formal_recovery_progress.json"
    run_config = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "noise": list(args.noise),
        "seed": args.seed,
        "methods": list(args.methods),
        "frequency_reference": (
            str(args.frequency_reference)
            if args.frequency_reference is not None
            else None
        ),
    }
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("config") != run_config:
            raise ValueError(
                "Existing recovery progress was created with different "
                "checkpoint/input/noise/seed/method settings."
            )
        all_runs = progress.get("runs", {})
    else:
        all_runs = {}

    def save_progress() -> None:
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"config": run_config, "runs": all_runs}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(progress_path)

    pretrained, diffusion_config = load_diffusion_checkpoint(args.checkpoint, device)
    model_config = pretrained.config
    dataset = load_graph_dataset(args.input, args.smiles_column)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_graphs,
    )
    frequency_modes = None
    if "frequency" in args.methods:
        if args.frequency_reference is None:
            raise ValueError(
                "--frequency-reference is required for the frequency baseline; "
                "use the clean pretraining corpus, never the recovery test set."
            )
        frequency_cache = args.output_dir / "frequency_modes.json"
        if args.resume and frequency_cache.exists():
            cached = json.loads(frequency_cache.read_text(encoding="utf-8"))
            if cached.get("reference") != str(args.frequency_reference):
                raise ValueError("Cached frequency modes use a different reference corpus")
            frequency_modes = (
                torch.tensor(cached["node_modes"], dtype=torch.long),
                int(cached["bond_mode"]),
            )
            print(f"loaded frequency modes from {frequency_cache}", flush=True)
        else:
            frequency_dataset = load_graph_dataset(
                args.frequency_reference, args.smiles_column
            )
            frequency_loader = DataLoader(
                frequency_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False,
                collate_fn=collate_graphs,
            )
            frequency_modes = compute_frequency_modes(
                frequency_loader, model_config
            )
            frequency_cache.write_text(
                json.dumps(
                    {
                        "reference": str(args.frequency_reference),
                        "node_modes": frequency_modes[0].tolist(),
                        "bond_mode": frequency_modes[1],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"fit frequency modes on clean reference "
                f"{args.frequency_reference}",
                flush=True,
            )
    models = {"pretrained": pretrained}
    if "random" in args.methods:
        torch.manual_seed(args.seed)
        models["random"] = BondAwareDiffusionModel(
            ModelConfig(**model_config.to_dict())
        ).to(device).eval()

    summaries = []
    for method in args.methods:
        model = models.get(method)
        method_runs = all_runs.setdefault(method, {})
        for noise in args.noise:
            key = f"{noise:.4f}"
            runs = list(method_runs.get(key, []))
            if len(runs) > args.repeats:
                runs = runs[: args.repeats]
            for repeat in range(len(runs), args.repeats):
                run = evaluate_formal_recovery_once(
                    model,
                    loader,
                    noise,
                    diffusion_config,
                    model_config,
                    device,
                    args.seed + repeat,
                    frequency_modes if method == "frequency" else None,
                )
                runs.append(run)
                method_runs[key] = runs
                save_progress()
                print(
                    f"checkpointed {method} noise={noise:.0%} "
                    f"repeat={repeat + 1}/{args.repeats}",
                    flush=True,
                )
            if args.resume and len(runs) == args.repeats:
                print(
                    f"complete {method} noise={noise:.0%}: "
                    f"{len(runs)}/{args.repeats} repeats",
                    flush=True,
                )
            summary = summarize_formal_runs(method, noise, runs)
            summaries.append(summary)
            print(
                f"{method:10s} noise={noise:.0%} "
                f"node_acc={summary['node_recovery_accuracy']:.4f} "
                f"exist_F1={summary['bond_existence_f1']:.4f} "
                f"type_F1={summary['bond_type_macro_f1']:.4f} "
                f"delta_valence={summary['valence_violation_delta']:+.2%} "
                f"affected_positive_substructure_F1="
                f"{summary['affected_positive_substructure_macro_f1']:.4f} "
                f"whole_graph_substructure_F1="
                f"{summary['substructure_macro_f1']:.4f}",
                flush=True,
            )
    save_formal_recovery(args.output_dir, summaries, all_runs)
    print(f"saved formal recovery benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
