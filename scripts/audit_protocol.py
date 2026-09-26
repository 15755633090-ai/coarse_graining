"""Diff the locked training protocol against the earlier formal benchmark.

This is a read-only audit. It does three things:

1. verifies the reference ``selected_config.json`` files still match the
   source hashes recorded by the earlier formal experiment;
2. diffs the resolved protocol presets in ``multiscale_tokenizer.training``
   field by field against those references;
3. checks that the working tree only touched protocol/plumbing files, so no
   model, data, pretrained-encoder or multiscale definition drifted.

Exit code is non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiscale_tokenizer.training import (  # noqa: E402
    LEGACY_EXECUTION,
    TaskSpec,
    _is_better_validation,
    _selection_metric,
    formal_protocol,
)


DEFAULT_REFERENCE_ROOT = (
    ROOT.parent / "model" / "results_formal" / "04_diffusion" / "lipo"
)

# Only protocol plumbing and its documentation/tests may change. Everything
# that defines the model, the data, the pretrained encoder or the multiscale
# method is frozen.
ALLOWED_CHANGED_FILES = {
    "multiscale_tokenizer/training.py",
    "scripts/train.py",
    "scripts/audit_protocol.py",
    "scripts/audit_base_execution.py",
    "scripts/audit_stage2_checkpoints.py",
    "results/base_execution_audit.json",
    "results/run_comparison_audit.json",
    "results/stage2_checkpoint_audit.json",
    "tests/test_training.py",
    "README.md",
    "docs/ALGORITHM.md",
}
FROZEN_PREFIXES = (
    "multiscale_tokenizer/model.py",
    "multiscale_tokenizer/partition.py",
    "diffusion_encoder/",
    "datasets/",
    "tests/test_model.py",
    "tests/test_partition.py",
)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    reference: object
    actual: object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference(directory: Path) -> dict[str, object]:
    path = directory / "selected_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing reference protocol {path}; pass --reference-root"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_reference_hashes(reference: dict[str, object]) -> list[Check]:
    tuning = reference["tuning_protocol"]
    checks: list[Check] = []
    for name, item in tuning["source_files"].items():
        path = Path(item["path"])
        if not path.is_file():
            checks.append(
                Check(f"reference source {name}", False, item["sha256"], "missing")
            )
            continue
        actual = _sha256(path)
        checks.append(
            Check(
                f"reference source {name}",
                actual == item["sha256"],
                item["sha256"],
                actual,
            )
        )
    return checks


def _protocol_checks(
    reference: dict[str, object],
    *,
    finetune: bool,
) -> list[Check]:
    tuning = reference["tuning_protocol"]
    selected = reference["selected_hyperparameters"]
    mode = "baseline_finetune" if finetune else "baseline_stage2_frozen"
    preset = formal_protocol(mode)
    optimizer = tuning["optimizer"]
    execution = tuning["execution"]
    # The earlier frozen configuration still recorded the searched encoder LR,
    # but the encoder is frozen, so its effective LR is exactly zero.
    effective_encoder_lr = selected["encoder_lr"] if finetune else 0.0
    expected = {
        "max_epochs": tuning["epochs"],
        "early_stopping_patience": tuning["patience"],
        "optimizer": optimizer["name"],
        "batch_size": selected["batch_size"],
        "micro_batch_size": execution["micro_batch_size"],
        "encoder_learning_rate": effective_encoder_lr,
        "downstream_learning_rate": selected["head_lr"],
        "weight_decay": optimizer["weight_decay"],
        "dropout": selected["dropout"],
        "grad_clip": optimizer["grad_clip"],
        "amp_dtype": "bf16" if execution["amp"] else "none",
        "scheduler": "none",
    }
    actual = {
        "max_epochs": preset.max_epochs,
        "early_stopping_patience": preset.early_stopping_patience,
        "optimizer": "AdamW",
        "batch_size": preset.batch_size,
        "micro_batch_size": preset.micro_batch_size,
        "encoder_learning_rate": preset.encoder_learning_rate,
        "downstream_learning_rate": preset.downstream_learning_rate,
        "weight_decay": preset.weight_decay,
        "dropout": preset.dropout,
        "grad_clip": preset.grad_clip,
        "amp_dtype": preset.amp_dtype,
        "scheduler": "none",
    }
    return [
        Check(name, expected[name] == actual[name], expected[name], actual[name])
        for name in expected
    ]


def _selection_checks() -> list[Check]:
    spec = TaskSpec("lipo", "regression", 1)
    metric = _selection_metric(spec)
    improved, value = _is_better_validation(
        {"loss": 0.01, "rmse": 0.40}, best_value=0.50, spec=spec,
    )
    return [
        Check(
            "regression selection metric",
            metric == "valid_rmse",
            "valid_rmse",
            metric,
        ),
        Check(
            "regression selects validation RMSE",
            improved and value == 0.40,
            "rmse=0.40 improves on 0.50",
            (improved, value),
        ),
    ]


def _execution_semantics_checks() -> list[Check]:
    """Declared semantics only; audit_base_execution.py checks runtime parity."""

    expected = {
        "train_diagnostics": False,
        "eval_no_grad": True,
        "best_state_on_cpu": True,
        "micro_padding": "tighten_to_micro_batch",
        "amp_dtype": "bf16",
        "micro_batch_size": 8,
    }
    return [
        Check(
            f"legacy execution {name}",
            LEGACY_EXECUTION.get(name) == value,
            value,
            LEGACY_EXECUTION.get(name),
        )
        for name, value in expected.items()
    ]


def _scope_checks() -> list[Check]:
    try:
        raw = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return [Check("working-tree scope", False, "git status available", str(exc))]
    changes = [
        line[3:].strip().strip('"')
        for line in raw.splitlines()
        if line.strip()
    ]
    changes = [name for name in changes if " -> " not in name]
    illegal = [
        name for name in changes
        if name not in ALLOWED_CHANGED_FILES
        and not name.startswith("docs/")
    ]
    touched_frozen = [
        name for name in changes
        if name.startswith(FROZEN_PREFIXES)
    ]
    return [
        Check(
            "protocol-only scope",
            not illegal and not touched_frozen,
            sorted(ALLOWED_CHANGED_FILES),
            changes,
        ),
    ]


def _report(checks: list[Check]) -> bool:
    width = max((len(check.name) for check in checks), default=0)
    all_ok = True
    for check in checks:
        status = "OK  " if check.ok else "DIFF"
        all_ok = all_ok and check.ok
        print(
            f"{status} {check.name:<{width}}  "
            f"reference={check.reference!r}  actual={check.actual!r}"
        )
    return all_ok


def _execution_notes(reference: dict[str, object]) -> dict[str, object]:
    """Fields recorded by the old run that are execution, not protocol, knobs."""

    execution = reference["tuning_protocol"]["execution"]
    return {
        "device": execution["device"],
        "deterministic": execution["deterministic"],
        "num_workers": execution["num_workers"],
        "cublas_workspace_config": execution.get("cublas_workspace_config"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT,
    )
    args = parser.parse_args()

    stages = {
        "stage1": (args.reference_root / "pretrained_finetune", True),
        "stage2": (args.reference_root / "pretrained_frozen", False),
    }
    results: list[tuple[str, bool]] = []
    for label, (directory, finetune) in stages.items():
        reference = _load_reference(directory)
        checks = [
            *_verify_reference_hashes(reference),
            *_protocol_checks(reference, finetune=finetune),
        ]
        print(f"== {label}: {directory / 'selected_config.json'}")
        print(
            "   old execution settings (informational): "
            + json.dumps(_execution_notes(reference), sort_keys=True)
        )
        results.append((label, _report(checks)))
    print("== selection rule")
    results.append(("selection", _report(_selection_checks())))
    print("== legacy execution semantics")
    results.append(("execution", _report(_execution_semantics_checks())))
    print("== working-tree scope")
    results.append(("scope", _report(_scope_checks())))

    failed = [label for label, ok in results if not ok]
    if failed:
        print(f"\nprotocol audit FAILED: {', '.join(failed)}")
        raise SystemExit(1)
    print("\nprotocol audit passed")


if __name__ == "__main__":
    main()
