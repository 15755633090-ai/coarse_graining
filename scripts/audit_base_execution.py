"""Compare original and extracted Base initialization without training."""
import hashlib
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiscale_tokenizer.training import (
    build_property_model, TASK_SPECS, TargetScaler, build_optimizer,
    _run_epoch, _make_loader, _slice_property_batch,
)
from diffusion_encoder.data import OGBMoleculePropertyDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-epochs", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT.parent / "model/results_formal/04_diffusion/lipo/pretrained_finetune/seed_0/run_config.json").read_text())
    protocol = config["tuning_protocol"]
    source = Path(protocol["source_files"]["property_prediction.py"]["path"])
    sys.path.insert(0, str(source.parent.parent))
    from bond_diffusion.property_prediction import create_property_model
    torch.manual_seed(0)
    old = create_property_model("pretrained_finetune", 1, protocol["checkpoint"]["path"], torch.device("cpu"), 0.3)
    old_rng = torch.get_rng_state().clone()
    new = build_property_model(spec=TASK_SPECS["lipo"], experiment_mode="baseline_finetune", model_seed=0, partition_seed=100000, dropout=0.3, device=torch.device("cpu"))
    encoder_equal = all(torch.equal(v, old.encoder.state_dict()[k]) for k, v in new.encoder.state_dict().items())
    heads = {k: {"equal": torch.equal(v, old.head.state_dict()[k]), "max_abs_difference": (v-old.head.state_dict()[k]).abs().max().item()} for k,v in new.prediction_head.state_dict().items()}
    report = {
        "reference_property_source_hash_matches": hashlib.sha256(source.read_bytes()).hexdigest() == protocol["source_files"]["property_prediction.py"]["sha256"],
        "pretrained_encoder_tensors_equal": encoder_equal,
        "rng_after_construction_equal": torch.equal(old_rng, torch.get_rng_state()),
        "head_initialization": heads,
        "limitations": "Initialization and one real training batch only; full-run metrics not verified."
    }
    from bond_diffusion import property_prediction as legacy
    from downstream_benchmark import _slice_property_batch as old_slice
    from torch.utils.data import DataLoader, Subset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old.to(device)
    new.to(device)
    dataset = OGBMoleculePropertyDataset(ROOT / "datasets/lipo", split="train")
    old_dataset = legacy.PropertyDataset(ROOT / "datasets/lipo/mapping/mol.csv.gz", legacy.TASK_SPECS["lipo"])
    old_loader = DataLoader(Subset(old_dataset, list(dataset.sample_ids)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0), collate_fn=legacy.collate_property_batch)
    new_loader = _make_loader(dataset, batch_size=64, shuffle=True, num_workers=0, base_partition_seed=100000, epoch=0, training=True, generator=torch.Generator().manual_seed(0), use_partitions=False)
    a, b = next(iter(old_loader)), next(iter(new_loader))
    report["first_batch_equal"] = all(torch.equal(getattr(a.graph,k), getattr(b.graph,k)) for k in ("node_features", "bonds", "node_mask")) and torch.equal(a.targets,b.targets)
    scaler = TargetScaler.fit(dataset)
    old_scaler = legacy.TargetScaler.fit(old_dataset.targets[dataset.sample_ids], old_dataset.target_mask[dataset.sample_ids])
    report["scaler_equal"] = torch.equal(scaler.mean, old_scaler.center) and torch.equal(scaler.std, old_scaler.scale)
    old_opt = torch.optim.AdamW([{"params": old.head.parameters(), "lr": 5e-4}, {"params": [p for p in old.encoder.parameters() if p.requires_grad], "lr": 1e-4}], weight_decay=1e-5)
    new_opt = build_optimizer(new, encoder_learning_rate=1e-4, downstream_learning_rate=5e-4, weight_decay=1e-5)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    old.train()
    old_opt.zero_grad(set_to_none=True)
    outputs = []
    for start in range(0, len(a.targets), 8):
        micro = old_slice(a, start, start+8).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = old(micro.graph)
            loss = legacy.property_loss(output, micro.targets, micro.target_mask, legacy.TASK_SPECS["lipo"], old_scaler, None)
            loss = loss * (float(micro.target_mask.sum()) / float(a.target_mask.sum()))
        outputs.append(output.detach())
        loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in old.parameters() if p.requires_grad], 5.0)
    old_opt.step()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    new_outputs = []
    hook = new.prediction_head.register_forward_hook(lambda module, inputs, output: new_outputs.append(output.detach()))
    _run_epoch(new, [b], device=device, spec=TASK_SPECS["lipo"], scaler=scaler, base_partition_seed=100000, epoch=0, training=True, optimizer=new_opt, micro_batch_size=8, grad_clip=5.0, amp_dtype="bf16", collect_diagnostics=False)
    hook.remove()
    report["device"] = str(device)
    report["training_predictions_equal"] = all(torch.equal(x,y) for x,y in zip(outputs,new_outputs)) and len(outputs)==len(new_outputs)
    pairs = [(p, dict(old.named_parameters())[name.replace("prediction_head.", "head.")]) for name,p in new.named_parameters()]
    report["clipped_gradients_equal"] = all(torch.equal(p.grad,q.grad) for p,q in pairs)
    report["parameters_after_step_equal"] = all(torch.equal(p,q) for p,q in pairs)
    old.eval()
    new.eval()
    with torch.no_grad():
        ma, mb = old_slice(a,0,8).to(device), _slice_property_batch(b,0,8).to(device)
        report["fp32_eval_predictions_equal"] = torch.equal(old(ma.graph),new(mb.graph.node_features,mb.graph.bonds,mb.graph.node_mask).prediction)
    if args.replay_epochs:
        # Fresh initialization and loader generators, exactly as in run_single.
        del old, new, old_opt, new_opt, pairs
        model = build_property_model(spec=TASK_SPECS["lipo"], experiment_mode="baseline_finetune", model_seed=0, partition_seed=100000, dropout=0.3, device=device)
        optimizer = build_optimizer(model, encoder_learning_rate=1e-4, downstream_learning_rate=5e-4, weight_decay=1e-5)
        train_loader = _make_loader(dataset, batch_size=64, shuffle=True, num_workers=0, base_partition_seed=100000, epoch=0, training=True, generator=torch.Generator().manual_seed(0), use_partitions=False)
        valid = OGBMoleculePropertyDataset(ROOT / "datasets/lipo", split="valid")
        valid_loader = _make_loader(valid, batch_size=64, shuffle=False, num_workers=0, base_partition_seed=100000, epoch=0, training=False, generator=torch.Generator().manual_seed(10000), use_partitions=False)
        expected = json.loads((ROOT.parent / "model/results_formal/04_diffusion/lipo/pretrained_finetune/seed_0/history.json").read_text())
        replay = []
        for epoch in range(args.replay_epochs):
            _run_epoch(model, train_loader, device=device, spec=TASK_SPECS["lipo"], scaler=scaler, base_partition_seed=100000, epoch=epoch, training=True, optimizer=optimizer, micro_batch_size=8, grad_clip=5, amp_dtype="bf16", collect_diagnostics=False)
            metrics = _run_epoch(model, valid_loader, device=device, spec=TASK_SPECS["lipo"], scaler=scaler, base_partition_seed=100000, epoch=0, training=False, amp_dtype="bf16")
            target = expected[epoch]["valid_metrics"]["rmse"]
            replay.append({"epoch_one_based": epoch+1, "reference_rmse": target, "actual_rmse": metrics["rmse"], "absolute_difference": abs(target-metrics["rmse"])})
            print(json.dumps(replay[-1]), flush=True)
        report["replay"] = replay
        report["replay_matches_within_1e_6"] = all(r["absolute_difference"] <= 1e-6 for r in replay)
        report["limitations"] = "Short training replay only; complete training and final test not run."
    text = json.dumps(report, indent=2)
    (ROOT / "results/base_execution_audit.json").write_text(text, encoding="utf-8")
    print(text)
    if not all(v for k,v in report.items() if isinstance(v,bool)) or not all(v["equal"] for v in heads.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
