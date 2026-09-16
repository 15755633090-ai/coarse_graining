"""Materialize the minimal, hash-locked input bundle for server jobs."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import run_lipo_formal as formal


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    options = parser.parse_args()

    destination = options.destination.resolve()
    legacy, args, protocol, baseline = formal.load_protocol(formal.PROJECT)
    selection_path = baseline / "lipo/pretrained_frozen/selected_config.json"
    selected = formal.read_json(selection_path)
    frozen_protocol = selected["tuning_protocol"]
    for key in ("split_sha256", "checkpoint", "pretrain_smiles", "source_files"):
        if frozen_protocol[key] != protocol[key]:
            raise ValueError(f"Frozen protocol differs from formal protocol: {key}")

    assets = destination / "assets"
    legacy_root = Path(frozen_protocol["source_files"]["downstream_benchmark.py"]["path"]).parent
    for source in (legacy_root / "downstream_benchmark.py", *(legacy_root / "bond_diffusion").glob("*.py")):
        _copy(source, assets / "legacy" / source.relative_to(legacy_root))
    _copy(args.checkpoint, assets / "encoder_best.pt")
    _copy(args.pretrain_smiles, assets / "ogb_pretrain_clean.csv")
    _copy(selection_path, assets / "selected_config.json")
    _copy(args.search_space, assets / "search_spaces.json")

    source_data = args.data_root / "ogbg_mollipo"
    for source in source_data.rglob("*"):
        if source.is_file():
            _copy(source, assets / "data" / source.relative_to(args.data_root))

    mother = formal.PROJECT / "model/results_formal/05_coarse_gnn/frozen_mechanism"
    preflight = formal.read_json(mother / "preflight.json")
    cache_source = Path(preflight["feature_cache"]["path"])
    _copy(cache_source, assets / "frozen_encoder_features.pt")

    references = []
    for variant in ("region_only", "base_coarse"):
        for seed in range(3):
            path = mother / "lipo" / variant / f"seed_{seed}" / "result.json"
            result = formal.read_json(path)
            score = (result.get("validation_metrics") or {}).get("rmse")
            if result.get("mode") != variant or result.get("seed") != seed or score is None:
                raise ValueError(f"Invalid formal reference: {path}")
            references.append({
                "mode": variant,
                "seed": seed,
                "validation_rmse": score,
            })
    formal.write_json(assets / "reference_validation.json", references)

    code_paths = [
        formal.ROOT / "run_lipo_formal.py",
        formal.ROOT / "frozen_features.py",
        *(formal.ROOT / "coarse_gnn").glob("*.py"),
        formal.ROOT / "scripts/readout_ablation/models.py",
        *(formal.ROOT / "scripts/server_batch").glob("*.py"),
    ]

    manifest = {
        "schema_version": 1,
        "purpose": "portable frozen size-weighted readout batch",
        "selected_hyperparameters": selected["selected_hyperparameters"],
        "frozen_protocol": frozen_protocol,
        "feature_cache": {
            "source_identity": preflight["feature_cache"],
            "bundle_file": "assets/frozen_encoder_features.pt",
            "sha256": legacy.file_identity(assets / "frozen_encoder_features.pt")["sha256"],
        },
        "protocol_scope": {
            "seeds": [0, 1, 2],
            "status": "exploratory_three_seed_validation_only",
            "not_a_replacement_for": "five_seed_formal_protocol",
        },
        "code": {
            "required_commit": _commit(formal.ROOT),
            "files": {
                str(path.relative_to(formal.ROOT)).replace("\\", "/"): legacy.file_identity(path)
                for path in sorted(code_paths)
            },
        },
        "reference_validation": "assets/reference_validation.json",
        "files": {
            str(path.relative_to(destination)).replace("\\", "/"): legacy.file_identity(path)
            for path in sorted(assets.rglob("*")) if path.is_file()
        },
    }
    formal.write_json(destination / "manifest.json", manifest)
    print(f"Portable bundle created: {destination}", flush=True)


if __name__ == "__main__":
    main()
