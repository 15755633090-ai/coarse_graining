"""Cohort guards, dependency provenance and summaries rebuilt from completed runs."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid

import pandas as pd
import torch


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


@contextmanager
def output_session(output, dataset):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / f".{dataset}.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Output session is locked: {lock}; use another directory or inspect the prior process") from None
    try:
        with handle:
            json.dump({"pid": os.getpid()}, handle)
        yield
    finally:
        lock.unlink()


def _git(directory, *arguments):
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), *arguments], stderr=subprocess.DEVNULL, timeout=10
        ).decode("utf-8").strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance(root):
    module = Path(__file__).parent
    paths = set(module.glob("*.py"))
    paths.update(root / "model" / n for n in (
        "benchmark.py", "gine.py", "attentivefp.py", "pna.py", "morgan.py"
    ))
    paths.update((root / "扩散GNN/01_扩散预训练/bond_diffusion").glob("*.py"))
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    packages = {}
    for name in ("torch", "torch-geometric", "ogb", "rdkit", "numpy", "pandas", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # A workspace export may sit beside a Git checkout. Never label its HEAD as
    # the running source revision without checking every recorded source blob.
    checkout = root if _git(root, "rev-parse", "--show-toplevel") else root / "coarse_graining1"
    head = _git(checkout, "rev-parse", "HEAD")
    published = _git(checkout, "rev-parse", "refs/remotes/origin/main")
    reference = published or head
    matches = False
    if reference:
        tree = _git(checkout, "-c", "core.quotePath=false", "ls-tree", "-r", reference, "--", *sources)
        objects = {}
        for line in (tree or "").splitlines():
            fields, name = line.split("\t", 1)
            objects[name] = fields.split()[2]
        matches = all(objects.get(name) == hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest() for name in sources for raw in [(root / name).read_bytes()])
    return {
        "source_sha256": sources,
        "software": {"python": platform.python_version(), "packages": packages,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "git": {"checkout": str(checkout) if head else None, "checkout_head_sha": head,
                "repository_commit_sha": reference, "source_matches_commit": matches,
                "reference_only": not matches},
    }


def prepare_cohort(args, protocol):
    """Validate everything before writing an audit or starting any requested run."""
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate variants or seeds")
    for seed in args.seeds:
        for variant in args.variants:
            folder = args.output_dir / args.dataset / variant / f"seed_{seed}"
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"Run exists; choose new seeds/variants or another output directory: {folder}")
    path = args.output_dir / f"{args.dataset}_protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Output cohort protocol differs (data, splits, software, source or training settings); use another output directory")
    else:
        if (args.output_dir / f"{args.dataset}_partition_audit.json").exists() or any(
            (args.output_dir / args.dataset).glob("*/seed_*/config.json")
        ):
            raise ValueError("Legacy outputs lack a cohort protocol; use another output directory")
        write_json(path, protocol)
    return fingerprint(protocol)


def save_partition_audit(output, dataset, audit):
    path = output / f"{dataset}_partition_audit.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != audit:
            raise ValueError("Partition audit differs; use another output directory")
    else:
        write_json(path, audit)


def refresh_summaries(output, dataset, protocol_sha256):
    rows, incomplete = [], []
    for config_path in sorted((output / dataset).glob("*/seed_*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"Incompatible run in output cohort: {config_path}")
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            incomplete.append(config_path.parent.relative_to(output).as_posix())
            continue
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (row["variant"], row["seed"]) != (config["variant"], config["seed"]):
            raise ValueError(f"Result/config identity mismatch: {metrics_path}")
        if config_path.parent.name != f"seed_{row['seed']}" or config_path.parent.parent.name != row['variant']:
            raise ValueError(f"Result/path identity mismatch: {metrics_path}")
        rows.append(row)
    # All existing completed runs, including ones from earlier invocations.
    write_json(output / f"{dataset}_summary.json", rows)
    write_json(output / f"{dataset}_summary_status.json", {"completed_runs": len(rows), "incomplete_runs": incomplete})
    flat = [{"variant": r["variant"], "seed": r["seed"], "best_epoch": r["best_epoch"],
             **{f"valid_{k}": v for k, v in r["validation"].items()},
             **{f"test_{k}": v for k, v in r.get("test", {}).items()}} for r in rows]
    frame = pd.DataFrame(flat, columns=None if flat else ["variant", "seed", "best_epoch"])
    atomic_text(output / f"{dataset}_seeds.csv", frame.to_csv(index=False))
    metrics = [k for k in frame if k.startswith(("valid_", "test_"))]
    aggregate = frame.groupby("variant")[metrics].agg(["mean", "std", "count"]) if metrics else pd.DataFrame()
    atomic_text(output / f"{dataset}_aggregate.csv", aggregate.to_csv())
    return rows
