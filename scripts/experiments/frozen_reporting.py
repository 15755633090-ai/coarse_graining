"""Five-seed reporting and a narrowly audited reporting-only source revision."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import run_lipo_formal as formal


SEEDS = tuple(range(5))
MODES = ("region_only", "base_coarse")
# Verified from the pre-review Frozen entry point used by existing C/D runs.
ORIGINAL_SOURCE_SHA256 = "91680b36ca20352d2f8690e8ad76307a06347662897d8558b8f7778517e005ff"
TRAINING_AST_SHA256 = "9259fe714d586a59ad1649c7214551de3a50cff0931e09c2bfd7b2c9cb0330e9"
_COMPAT_IMPORT = "from scripts.experiments.frozen_reporting import preserve_training_manifest"
_COMPAT_CALL = "config = preserve_training_manifest(legacy, config, args.output_dir)"
_SUMMARY_WRAPPER = """def summarize(args, baseline):
    from scripts.experiments.frozen_reporting import summarize_frozen
    return summarize_frozen(args, baseline)
"""
_ORIGINAL_DOCSTRING = "Stage one: fixed encoder, matched hyperparameters, two models and seeds 0/1."
_CURRENT_DOCSTRING = "Stage one: fixed encoder, matched hyperparameters, two models and seeds 0-4."


def _stable_ast(value):
    if isinstance(value, ast.AST):
        # Python 3.12 added empty type_params fields to ordinary functions/classes.
        # Omit only that empty metadata field, not nonempty generic declarations.
        return [type(value).__name__, [[name, _stable_ast(item)] for name, item in ast.iter_fields(value)
                                      if name != "type_params" or item]]
    if isinstance(value, list):
        return [_stable_ast(item) for item in value]
    return value


def training_ast_digest(source):
    """Validate the fixed summary wrapper before normalizing reporting-only edits."""
    tree = ast.parse(source)
    summaries = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "summarize"]
    expected = ast.parse(_SUMMARY_WRAPPER).body[0]
    if len(summaries) != 1 or _stable_ast(summaries[0]) != _stable_ast(expected):
        raise ValueError("Frozen summarize must exactly match the approved two-statement wrapper")
    tree.body.remove(summaries[0])
    # Permit only the reviewed seed-range documentation correction; normalize
    # that exact literal to retain the independently verified old training digest.
    if ast.get_docstring(tree, clean=False) not in (_ORIGINAL_DOCSTRING, _CURRENT_DOCSTRING):
        raise ValueError("Unexpected Frozen module docstring revision")
    tree.body[0].value.value = _ORIGINAL_DOCSTRING
    allowed = {ast.dump(ast.parse(line).body[0], include_attributes=False)
               for line in (_COMPAT_IMPORT, _COMPAT_CALL)}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            node.body = [item for item in node.body if ast.dump(item, include_attributes=False) not in allowed]
    payload = json.dumps(_stable_ast(tree), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_reporting_revision(previous, current, root=None):
    """Allow the known historical manifest only if every training AST node matches."""
    root = Path(root) if root is not None else formal.ROOT
    before, after = dict(previous), dict(current)
    old_files = before.pop("new_source_files")
    new_files = after.pop("new_source_files")
    if before != after or old_files.keys() != new_files.keys():
        raise ValueError("Frozen settings/source inventory changed beyond reporting; refusing compatibility")
    for name in old_files:
        if name != "run_lipo_frozen.py" and old_files[name] != new_files[name]:
            raise ValueError(f"Training source changed: {name}")
    old, new = old_files["run_lipo_frozen.py"], new_files["run_lipo_frozen.py"]
    if old["sha256"] != ORIGINAL_SOURCE_SHA256 or old["path"] != new["path"]:
        raise ValueError("Unknown Frozen source revision; refusing compatibility")
    source = root / "run_lipo_frozen.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != new["sha256"]:
        raise ValueError("Candidate Frozen source identity does not match disk")
    if training_ast_digest(source.read_text(encoding="utf-8")) != TRAINING_AST_SHA256:
        raise ValueError("Frozen training semantics changed; reporting-only compatibility is not allowed")


def preserve_training_manifest(legacy, current, output, *, write_audit=True):
    """Retain the old training contract, record actual reporting sources separately."""
    path = Path(output) / "run_config.json"
    if not path.exists():
        return current
    previous = formal.read_json(path)
    if previous == current:
        return previous
    verify_reporting_revision(previous, current)
    if write_audit:
        formal.write_json(Path(output) / "reporting_revision.json", dict(
            revision="five_seed_reporting_only", original_manifest_sha256=formal.digest(previous),
            original_manifest_preserved=True, training_ast_sha256=TRAINING_AST_SHA256,
            actual_source_files=current["new_source_files"],
            reporting_source=legacy.file_identity(Path(__file__)),
            note="Exact two-statement summary wrapper, reviewed seed-range docstring correction and compatibility hook only. No model/training/checkpoint migration.",
        ))
    return previous


def metric(row, split):
    value = (row.get(f"{split}_metrics") or {}).get("rmse")
    if value is None:
        return None
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Invalid {split} RMSE for {row.get('mode')}/seed_{row.get('seed')}")
    return value


def aggregate(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def summarize_frozen(args, baseline):
    output = Path(args.output_dir)
    runs, missing = [], []
    for mode in MODES:
        for seed in SEEDS:
            path = output / "lipo" / mode / f"seed_{seed}" / "result.json"
            if not path.exists():
                missing.append(f"{mode}/seed_{seed}")
                continue
            row = formal.read_json(path)
            if row.get("mode") != mode or row.get("seed") != seed:
                raise ValueError(f"Result identity mismatch: {path}")
            if metric(row, "validation") is None:
                raise ValueError(f"Missing selected-checkpoint validation metrics: {path}")
            metric(row, "test")
            runs.append(row)
    summary = []
    for mode in MODES:
        for split in ("validation", "test"):
            available = [(r["seed"], metric(r, split)) for r in runs if r["mode"] == mode and metric(r, split) is not None]
            summary.append(dict(mode=mode, metric_split=split, seeds=[s for s, _ in available],
                                **aggregate([value for _, value in available])))
    lookup = {(r["mode"], r["seed"]): r for r in runs}
    paired = []
    for seed in SEEDS:
        if not all((mode, seed) in lookup for mode in MODES):
            continue
        row = dict(seed=seed)
        for split, key in (("validation", "validation_rmse_base_minus_region"), ("test", "rmse_base_minus_region")):
            c, d = (metric(lookup[mode, seed], split) for mode in MODES)
            row[key] = d - c if c is not None and d is not None else None
        paired.append(row)
    paired_summary = {
        split: aggregate([r[key] for r in paired if r[key] is not None])
        for split, key in (("validation", "validation_rmse_base_minus_region"), ("test", "rmse_base_minus_region"))
    }
    # Historical atom baseline remains explicitly separate: it has a different
    # feature precision policy and may have fewer completed seeds.
    historical = []
    for seed in SEEDS:
        path = Path(baseline) / "lipo/pretrained_frozen" / f"seed_{seed}" / "result.json"
        if path.exists():
            row = formal.read_json(path)
            historical.append(dict(seed=seed, validation_rmse=metric(row, "validation"), test_rmse=metric(row, "test")))
    report = dict(schema_version=2, eligible_seeds=list(SEEDS), complete=not missing,
                  missing_runs=missing, summary=summary, paired=paired, paired_summary=paired_summary,
                  reporting_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (Path(__file__), formal.ROOT / "run_lipo_frozen.py")},
                  primary_development_split="validation", historical_baseline=historical,
                  interpretation="Use validation for method development. Existing C/D test scores are retrospective reporting only; historical atom scores are not a matched-cache control.")
    formal.write_json(output / "downstream_results.json", dict(runs=runs, summary=summary))
    formal.write_json(output / "comparison.json", report)
    with (output / "comparison_seed_0_4.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("mode", "metric_split", "seeds", "n", "mean", "std"))
        writer.writeheader()
        writer.writerows(dict(row, seeds=";".join(map(str, row["seeds"]))) for row in summary)
    return report
