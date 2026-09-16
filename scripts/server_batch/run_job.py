"""Run one portable frozen size-weighted readout job on one CUDA device."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import os
import sys
from pathlib import Path

import torch

import run_lipo_formal as formal
from coarse_gnn import TopologyCache
from coarse_gnn.prepared_data import PreparedPropertyDataset
from frozen_features import FrozenDataset, frozen_tools
from scripts.readout_ablation import run_size_weighted as readout
from scripts.readout_ablation.models import TRAIN_VARIANTS, make_factory


def _identity(legacy, path: Path) -> dict:
    return legacy.file_identity(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_code(manifest: dict) -> None:
    root = formal.ROOT
    code = manifest.get("code") or {}
    if code.get("mode") != "self_contained_hash_locked" or code.get("root") != "code":
        raise ValueError("Experiment package does not contain a self-contained source lock")
    expected_files = code.get("files", {})
    for relative, expected in expected_files.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected["sha256"]:
            raise ValueError(f"Required project source differs from bundle: {relative}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    unexpected = sorted(actual_files - set(expected_files))
    if unexpected:
        raise ValueError(f"Experiment package contains unexpected source files: {unexpected}")
    launcher = manifest.get("launcher") or {}
    launcher_path = root.parent / launcher.get("path", "")
    if not launcher_path.is_file() or _sha256(launcher_path) != launcher.get("sha256"):
        raise ValueError("Server launcher differs from bundle manifest")


def load_context(bundle: Path):
    manifest = formal.read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != 4:
        raise ValueError("Unsupported portable bundle manifest")
    scope = manifest.get("protocol_scope")
    if scope != {
        "seeds": [0, 1, 2],
        "status": "exploratory_three_seed_validation_only",
        "not_a_replacement_for": "five_seed_formal_protocol",
    }:
        raise ValueError("Unexpected server protocol scope")
    execution = manifest.get("server_execution")
    expected_execution = {
        "device": "cuda:0",
        "amp": manifest["frozen_protocol"]["execution"]["amp"],
        "deterministic": manifest["frozen_protocol"]["execution"]["deterministic"],
        "micro_batch_size": manifest["frozen_protocol"]["execution"]["micro_batch_size"],
        "num_workers": manifest["frozen_protocol"]["execution"]["num_workers"],
        "cpu_threads": 8,
        "cublas_workspace_config": manifest["frozen_protocol"]["execution"]["cublas_workspace_config"],
    }
    if execution != expected_execution:
        raise ValueError("Server execution does not match the locked mother execution protocol")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = execution["cublas_workspace_config"]
    _verify_code(manifest)
    assets = bundle / "assets"
    for relative, expected in manifest["files"].items():
        actual = _sha256(assets.parent / relative)
        if actual != expected["sha256"]:
            raise ValueError(f"Bundle file changed: {relative}")

    legacy_root = assets / "legacy"
    sys.path.insert(0, str(legacy_root))
    legacy = importlib.import_module("downstream_benchmark")
    protocol = manifest["frozen_protocol"]
    for name, expected in protocol["source_files"].items():
        copied = legacy_root / ("downstream_benchmark.py" if name == "downstream_benchmark.py" else f"bond_diffusion/{name}")
        if _identity(legacy, copied)["sha256"] != expected["sha256"]:
            raise ValueError(f"Original source hash changed: {name}")
    with formal.overrides(sys, argv=[str(legacy_root / "downstream_benchmark.py")]):
        args = legacy.parse_args()
    for key in ("epochs", "patience", "tuning_seed", "overlap_policy"):
        setattr(args, key, protocol[key])
    args.weight_decay = protocol["optimizer"]["weight_decay"]
    args.grad_clip = protocol["optimizer"]["grad_clip"]
    args.data_root = assets / "data"
    args.checkpoint = assets / "encoder_best.pt"
    args.pretrain_smiles = assets / "ogb_pretrain_clean.csv"
    args.search_space = assets / "search_spaces.json"
    args.device = execution["device"]
    args.micro_batch_size = execution["micro_batch_size"]
    args.num_workers = execution["num_workers"]
    args.amp = execution["amp"]
    args.deterministic = execution["deterministic"]
    for key, value in manifest["selected_hyperparameters"].items():
        setattr(args, key, value)
    torch.set_num_threads(execution["cpu_threads"])
    torch.use_deterministic_algorithms(execution["deterministic"])

    source, splits, spec = legacy.build_property_data(args.data_root, "lipo")
    if legacy.split_sha256(source, splits) != protocol["split_sha256"]:
        raise ValueError("Portable Lipo data does not match the locked split")
    cache = TopologyCache(max_memory_entries=8192)
    prepared = PreparedPropertyDataset(source, legacy.collate_property_batch, cache, formal.STRUCTURE)
    saved = torch.load(assets / "frozen_encoder_features.pt", map_location="cpu", weights_only=True)
    identity = saved.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Frozen feature cache has no identity")
    if identity["checkpoint"]["sha256"] != protocol["checkpoint"]["sha256"]:
        raise ValueError("Frozen feature cache checkpoint identity mismatch")
    if identity.get("structure") != formal.STRUCTURE.__dict__:
        raise ValueError("Frozen feature cache coarsening configuration mismatch")
    if identity.get("encoder_mode") != "eval_no_grad" or identity.get("precision") != "float32_autocast_disabled":
        raise ValueError("Frozen feature cache encoder mode or precision mismatch")
    expected_fingerprints = [prepared[index][3].input_fingerprint for index in range(len(prepared))]
    if identity.get("graph_fingerprints") != expected_fingerprints:
        raise ValueError("Frozen feature cache graph fingerprints/order mismatch")
    if len(saved["features"]) != len(prepared):
        raise ValueError("Frozen feature cache sample count mismatch")
    for index, nodes in enumerate(saved["features"]):
        if tuple(nodes.shape) != (len(prepared[index][3].owner), 128) or nodes.dtype != torch.float32:
            raise ValueError(f"Invalid frozen feature tensor: {index}")
        if not torch.isfinite(nodes).all():
            raise ValueError(f"Nonfinite frozen feature tensor: {index}")
    data = FrozenDataset(prepared, saved["features"])
    factory = make_factory(legacy.create_property_model, cache)
    tools = frozen_tools(legacy)
    return manifest, legacy, args, data, splits, spec, factory, tools


def run(options: argparse.Namespace) -> None:
    bundle = options.bundle.resolve()
    output = options.output_dir.resolve()
    manifest, legacy, args, data, splits, spec, factory, tools = load_context(bundle)
    args.output_dir = output
    parameters = readout.parameter_report(factory, args, spec.num_tasks, legacy)
    config = {
        "stage": "portable_frozen_size_weighted_batch",
        "schema_version": 1,
        "bundle_manifest_sha256": formal.digest(manifest),
        "protocol_scope": manifest["protocol_scope"],
        "evaluation_policy": dict(readout.EVALUATION_POLICY),
        "variants": [options.variant],
        "seed": options.seed,
        "execution": dict(manifest["server_execution"]),
        "models": parameters,
        "change": "graph_pool only: mean -> size_weighted_mean",
    }
    output.mkdir(parents=True, exist_ok=True)
    formal.write_json(output / "run_config.json", config)
    report = readout.preflight(legacy, args, data, splits, spec, factory, tools, args.device)
    formal.write_json(output / "preflight.json", report)
    if options.action == "prepare":
        print(f"Prepared portable job: {output}", flush=True)
        return
    run_args = copy.copy(args)
    payload = dict(config, mode=options.variant)
    run_args.tuning_protocol = {"payload": payload, "sha256": formal.digest(payload)}
    directory = output / "lipo" / options.variant / f"seed_{options.seed}"
    expected = legacy.single_run_config(spec, options.variant, options.seed, "report", {key: getattr(args, key) for key in ("batch_size", "encoder_lr", "head_lr", "dropout")}, run_args.tuning_protocol)
    legacy.initialize_single_run_config(directory, expected)
    if (directory / "result.json").is_file():
        result = formal.read_json(directory / "result.json")
        if result.get("test_metrics") is not None or (directory / "test_predictions.csv").exists():
            raise ValueError("Refusing to reuse a test-evaluated result")
        print(f"Already complete: {options.variant} seed={options.seed}", flush=True)
        return
    with formal.overrides(legacy, create_property_model=factory, **tools):
        legacy.run_single(run_args, data, splits, spec, options.variant, options.seed, torch.device(args.device), directory, evaluate_test=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=TRAIN_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=range(3), required=True)
    parser.add_argument("--action", choices=("prepare", "train"), default="train")
    options = parser.parse_args()
    run(options)


if __name__ == "__main__":
    main()
