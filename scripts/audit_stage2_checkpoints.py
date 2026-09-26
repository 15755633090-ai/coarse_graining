"""Read-only checks of the completed legacyexec Stage-2 checkpoints."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiscale_tokenizer.training import load_property_model


def main():
    source = torch.load(ROOT / "runs/lipo_formal_stage1_seed0_legacyexec/best.pt", map_location="cpu", weights_only=False)
    encoder = {k: v for k, v in source["model_state"].items() if k.startswith("encoder.")}
    report = []
    for path in sorted((ROOT / "runs").glob("lipo_formal_stage2*_legacyexec/best.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        last = torch.load(path.with_name("last.pt"), map_location="cpu", weights_only=False)
        model, _, _ = load_property_model(path)
        model.train()
        checks = {
            "encoder_best_equals_source": all(torch.equal(v, saved["model_state"][k]) for k, v in encoder.items()),
            "encoder_last_equals_source": all(torch.equal(v, last["last_model_state"][k]) for k, v in encoder.items()),
            "encoder_eval_after_train": all(not m.training for m in model.encoder.modules()),
            "encoder_all_frozen": all(not p.requires_grad for p in model.encoder.parameters()),
            "scaler_equals_stage1": all(torch.equal(v, source["target_scaler"][k]) for k, v in saved["target_scaler"].items()),
            "history_best_matches": saved["best_epoch"] == min(saved["history"], key=lambda r: r["valid_rmse"])["epoch"],
        }
        report.append({"run": path.parent.name, "epochs": len(saved["history"]), "best_epoch_zero_based": saved["best_epoch"], "checks": checks})
    text = json.dumps(report, indent=2)
    (ROOT / "results/stage2_checkpoint_audit.json").write_text(text, encoding="utf-8")
    print(text)
    if len(report) != 2 or not all(all(r["checks"].values()) for r in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
