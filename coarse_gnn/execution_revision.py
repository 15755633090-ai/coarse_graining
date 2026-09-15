"""Explicit, audited implementation revision without changing experiment knobs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def hashes(files):
    return {key.replace("\\", "/"): item["sha256"] for key, item in files.items()}


class ExecutionRevision:
    def __init__(self, root, output, current_files):
        self.root, self.output, self.current_files = Path(root), Path(output), current_files
        self.audit_path = self.root / "outputs/performance/packed_audit.json"
        self.audit = read(self.audit_path)
        if not self.audit.get("passed") or self.audit.get("source_hashes") != hashes(current_files):
            raise ValueError("Packed execution requires a passing verify_packed.py audit of the current source")
        reference = read(self.root / "outputs/performance/reference/manifest.json")
        self.reference = {key.replace("\\", "/"): value for key, value in reference.items()}
        self.entry = dict(execution="packed_v1", source_files=current_files,
                          audit_file=str(self.audit_path), audit_sha256=sha256(self.audit_path),
                          numerical_contract="FP32 close; BF16 accumulation rounding differs; dropout RNG order preserved",
                          optimizer_and_model_state_layout="unchanged")

    def compatible(self, previous, current):
        before, after = dict(previous), dict(current)
        old_files = before.pop("new_source_files", None)
        new_files = after.pop("new_source_files", None)
        if before != after or old_files is None or new_files is None:
            raise ValueError("Experiment settings differ beyond implementation provenance; refusing migration")
        if hashes(new_files) != hashes(self.current_files):
            raise ValueError("Unexpected current implementation identity")
        old_hashes = hashes(old_files)
        if old_hashes not in (self.reference, hashes(self.current_files)):
            raise ValueError("Old run does not match the audited pre-optimization source snapshot")

    def manifest(self, current):
        path = self.output / "run_config.json"
        if not path.exists():
            return current
        previous = read(path)
        if previous != current:
            self.compatible(previous, current)
            journal = self.output / "execution_revision.json"
            write(journal, dict(**self.entry, original_source_files=previous["new_source_files"],
                                experiment_manifest_preserved=True))
        return previous

    def protocol(self, current, variant):
        directory = self.output / "lipo" / variant
        selection = directory / "selected_config.json"
        candidates = ([selection] if selection.exists() else []) + sorted(directory.glob("*/seed_*/trial_*/run_config.json")) + sorted(directory.glob("seed_*/run_config.json"))
        if not candidates:
            return current
        previous = read(candidates[0])
        if previous["tuning_protocol"] != current["payload"]:
            self.compatible(previous["tuning_protocol"], current["payload"])
        return dict(payload=previous["tuning_protocol"], sha256=previous["tuning_protocol_sha256"])

    def run_wrapper(self, original):
        def run(args, dataset, splits, spec, mode, seed, device, run_dir=None, evaluate_test=True, role="report"):
            directory = run_dir or args.output_dir / spec.name / mode / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "execution_history.json"
            entries = read(path) if path.exists() else []
            resume = directory / "resume.pt"
            entries.append(dict(**self.entry, started_at=datetime.now(timezone.utc).isoformat(),
                                role=role, seed=seed,
                                resume_sha256=sha256(resume) if resume.exists() else None))
            write(path, entries)
            print(f"start {mode} {role} seed={seed}: packed_v1; resume={resume.exists()}", flush=True)
            return original(args, dataset, splits, spec, mode, seed, device, run_dir, evaluate_test, role)
        return run
