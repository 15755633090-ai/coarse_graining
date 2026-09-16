"""Materialize the minimal, hash-locked input bundle for server jobs."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import run_lipo_formal as formal
import run_lipo_frozen as frozen
from scripts.readout_ablation import run_size_weighted as readout


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _reference_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"], cwd=root,
        text=True, capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _prepare_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Bundle destination is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"Bundle destination must be empty: {destination}")
    else:
        destination.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    options = parser.parse_args()

    destination = options.destination.resolve()
    _prepare_destination(destination)
    legacy, args, protocol, baseline = formal.load_protocol(formal.PROJECT)
    selection_path = baseline / "lipo/pretrained_frozen/selected_config.json"
    selected = formal.read_json(selection_path)
    frozen_protocol = selected["tuning_protocol"]
    for key in ("split_sha256", "checkpoint", "pretrain_smiles", "source_files"):
        if frozen_protocol[key] != protocol[key]:
            raise ValueError(f"Frozen protocol differs from formal protocol: {key}")

    # Reuse the same mother-manifest validation as the local readout runner.  This
    # prevents a bundle from silently mixing selected hyperparameters with a
    # different frozen C/D mother experiment.
    frozen_legacy, frozen_args, frozen_protocol_args, _, selection = frozen.load_protocol(formal.PROJECT)
    mother_manifest = readout.verify_mother(
        frozen_legacy, frozen_args, frozen_protocol_args, selection, frozen_args.output_dir
    )
    effective_keys = tuple(mother_manifest["fixed_hyperparameters"])
    selected_effective = {key: selected["selected_hyperparameters"][key] for key in effective_keys}
    if mother_manifest["fixed_hyperparameters"] != selected_effective:
        raise ValueError("Selected hyperparameters differ from the locked frozen C/D mother")
    if mother_manifest.get("encoder_lr_effective") != 0:
        raise ValueError("Frozen C/D mother must retain a zero effective encoder learning rate")
    mother_execution = frozen_protocol["execution"]
    server_execution = {
        "device": "cuda:0",
        "amp": mother_execution["amp"],
        "deterministic": mother_execution["deterministic"],
        "micro_batch_size": mother_execution["micro_batch_size"],
        "num_workers": mother_execution["num_workers"],
        "cpu_threads": 8,
        "cublas_workspace_config": mother_execution["cublas_workspace_config"],
    }

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
    if legacy.file_identity(cache_source)["sha256"] != preflight["feature_cache"]["file_sha256"]:
        raise ValueError("Mother frozen feature cache no longer matches its preflight identity")
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
        formal.ROOT / "run_lipo_frozen.py",
        formal.ROOT / "frozen_features.py",
        formal.ROOT / "requirements.txt",
        formal.ROOT / "diffusion/requirements.txt",
        *(formal.ROOT / "coarse_gnn").glob("*.py"),
        *(formal.ROOT / "diffusion/bond_diffusion").glob("*.py"),
        formal.ROOT / "scripts/__init__.py",
        formal.ROOT / "scripts/readout_ablation/__init__.py",
        formal.ROOT / "scripts/readout_ablation/models.py",
        formal.ROOT / "scripts/readout_ablation/run_size_weighted.py",
        formal.ROOT / "scripts/experiments/__init__.py",
        formal.ROOT / "scripts/experiments/frozen_reporting.py",
        *(formal.ROOT / "scripts/server_batch").glob("*.py"),
    ]
    for source in code_paths:
        _copy(source, destination / "code" / source.relative_to(formal.ROOT))

    protocol_scope = {
        "seeds": [0, 1, 2],
        "status": "exploratory_three_seed_validation_only",
        "not_a_replacement_for": "five_seed_formal_protocol",
    }
    reference_commit = _reference_commit(formal.ROOT)
    code_files = {
        str(path.relative_to(formal.ROOT)).replace("\\", "/"): legacy.file_identity(path)
        for path in sorted(code_paths)
    }
    asset_files = {
        str(path.relative_to(destination)).replace("\\", "/"): legacy.file_identity(path)
        for path in sorted(assets.rglob("*")) if path.is_file()
    }
    package_identity = {
        "selected_hyperparameters": selected["selected_hyperparameters"],
        "server_execution": server_execution,
        "protocol_scope": protocol_scope,
        "code_files": code_files,
        "asset_files": asset_files,
    }
    package_id = formal.digest(package_identity)[:12]
    launcher_source = formal.ROOT / "scripts/server_batch/run_experiment.cmd"
    launcher_text = launcher_source.read_text(encoding="utf-8")
    if launcher_text.count("__PACKAGE_ID__") != 1:
        raise ValueError("Server launcher must contain exactly one package-id placeholder")
    launcher_path = destination / "run_experiment.cmd"
    launcher_path.write_text(
        launcher_text.replace("__PACKAGE_ID__", package_id),
        encoding="utf-8",
        newline="",
    )

    manifest = {
        "schema_version": 4,
        "package_id": package_id,
        "purpose": "portable frozen size-weighted readout batch",
        "selected_hyperparameters": selected["selected_hyperparameters"],
        "frozen_protocol": frozen_protocol,
        "server_execution": server_execution,
        "feature_cache": {
            "source_identity": preflight["feature_cache"],
            "bundle_file": "assets/frozen_encoder_features.pt",
            "sha256": legacy.file_identity(assets / "frozen_encoder_features.pt")["sha256"],
        },
        "protocol_scope": protocol_scope,
        "code": {
            "mode": "self_contained_hash_locked",
            "root": "code",
            "reference_commit": reference_commit,
            "files": code_files,
        },
        "launcher": {
            "path": "run_experiment.cmd",
            **legacy.file_identity(launcher_path),
        },
        "reference_validation": "assets/reference_validation.json",
        "files": asset_files,
    }
    formal.write_json(destination / "manifest.json", manifest)
    print(f"Portable bundle created: {destination}", flush=True)


if __name__ == "__main__":
    main()
