"""Compare geometry reuse with the original branch on a validation batch."""
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_encoder.data import OGBMoleculePropertyDataset, collate_property_records
from multiscale_tokenizer.training import load_property_model, _partition_seeds, seed_everything


def main():
    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = ROOT / "runs/lipo_random_region_seed0_r2_v1/best.pt"
    model, _, _ = load_property_model(checkpoint, device=device)
    namespace = {}
    source = subprocess.check_output(
        ["git", "show", "d1a9e99:multiscale_tokenizer/random_regions.py"],
        cwd=ROOT, text=True, encoding="utf-8",
    )
    exec(compile(source, "original_random_regions.py", "exec"), namespace)
    legacy = namespace["RandomRegionBranch"](128, model.config.dropout).to(device)
    optimized = model.random_regions
    legacy.load_state_dict(optimized.state_dict(), strict=True)
    original_forward = legacy.forward
    legacy.forward = lambda *args, geometry=None, **kwargs: original_forward(*args, **kwargs)
    dataset = OGBMoleculePropertyDataset(ROOT / "datasets/lipo", split="valid")
    batch = collate_property_records([dataset[i] for i in range(32)]).to(device)
    seeds = _partition_seeds(batch.sample_ids, base_seed=100000, epoch=0, training=False)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def geometry_context(branch):
        # Original model prepared topology only inside each branch invocation.
        return (patch("multiscale_tokenizer.model.prepare_region_geometry", return_value=None)
                if branch is legacy else nullcontext())

    def run(branch, training):
        model.random_regions = branch
        model.train(training)
        model.zero_grad(set_to_none=True)
        seed_everything(123)
        with geometry_context(branch), torch.set_grad_enabled(training), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=training and device == "cuda"):
            prediction = model(batch.graph.node_features, batch.graph.bonds,
                               batch.graph.node_mask, seeds).prediction
            if training:
                prediction.square().mean().backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        return prediction.detach().clone(), gradients

    report = {"reference_commit": "d1a9e99", "device": device, "validation_samples": 32, "views": 5}
    for training in (False, True):
        left, left_grad = run(legacy, training)
        right, right_grad = run(optimized, training)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert left_grad.keys() == right_grad.keys()
        for name in left_grad:
            torch.testing.assert_close(left_grad[name], right_grad[name], atol=0, rtol=0)
        report["training" if training else "evaluation"] = {
            "max_prediction_difference": float((left - right).abs().max()),
            "gradient_tensors_compared": len(left_grad), "exact_match": True,
        }
    timings = {}
    for name, branch in (("original", legacy), ("shared_geometry", optimized)):
        model.random_regions = branch
        model.eval()
        with geometry_context(branch), torch.no_grad():
            for _ in range(2):
                model(batch.graph.node_features, batch.graph.bonds, batch.graph.node_mask, seeds)
            sync()
            start = time.perf_counter()
            for _ in range(10):
                model(batch.graph.node_features, batch.graph.bonds, batch.graph.node_mask, seeds)
            sync()
        timings[name] = (time.perf_counter() - start) / 10
    report["warm_batch_seconds"] = timings
    report["note"] = "Warm-cache timing on one validation batch; not a full-epoch benchmark. Original path skips the new outer geometry preparation. No test-set evaluation."
    output = ROOT / "results/random_regions_geometry_reuse_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
