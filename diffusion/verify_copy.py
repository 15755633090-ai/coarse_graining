"""One-time source equivalence and diffusion training verification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from bond_diffusion.config import DiffusionConfig
from bond_diffusion.data import collate_graphs
from bond_diffusion.diffusion import DiscreteGraphDiffusion
from bond_diffusion.losses import ChemicalDiffusionLoss
from bond_diffusion.trainer import export_encoder, load_encoder
from train import demo_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "copy_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for row in manifest["copied_files"]:
        if row["path"] in manifest["modified_after_copy"]:
            continue
        assert hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() == row["source_sha256"], row["path"]
        checked += 1
    checkpoint = root / "outputs/ogb_clean/best.pt"
    model = load_encoder(checkpoint)
    spec = importlib.util.spec_from_file_location("bond_diffusion.original_model", args.source / "bond_diffusion/model.py")
    import sys
    original_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original_module
    spec.loader.exec_module(original_module)
    original = original_module.BondAwareDiffusionModel(model.config).eval()
    original.load_state_dict(model.state_dict(), strict=True)
    batch = collate_graphs([demo_dataset()[0], demo_dataset()[1]])
    x, e, mask = batch.node_features, batch.bonds, batch.node_mask
    timestep = torch.tensor([0, 25])
    with torch.no_grad():
        node_diff = (model.encode_nodes(x, e, mask) - original.encode_nodes(x, e, mask)).abs().max().item()
        graph_diff = (model.encode(x, e, mask) - original.encode(x, e, mask)).abs().max().item()
        new_out, old_out = model(x, e, mask, timestep), original(x, e, mask, timestep)
        bond_diff = (new_out.bond_logits - old_out.bond_logits).abs().max().item()
    assert node_diff == graph_diff == bond_diff == 0
    assert not hasattr(model.layers[0], "forward_with_message_gate")
    assert not hasattr(model, "encode_node_layers")
    exported = root / "outputs/ogb_clean/encoder.pt"
    export_encoder(checkpoint, exported)
    with torch.no_grad():
        torch.testing.assert_close(load_encoder(exported).encode_nodes(x, e, mask), model.encode_nodes(x, e, mask), rtol=0, atol=0)
    model.train()
    diffusion = DiscreteGraphDiffusion(DiffusionConfig(), model.config)
    noisy = diffusion.corrupt(batch)
    output = model(noisy.node_features, noisy.bonds, mask, noisy.timesteps)
    loss = ChemicalDiffusionLoss()(output, batch, noisy)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.node_embeddings[0].weight.grad is not None
    assert torch.isfinite(model.node_embeddings[0].weight.grad).all()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    report = {"status": "passed", "unchanged_files_sha256_verified": checked,
              "node_embedding_max_abs_diff": node_diff, "legacy_graph_embedding_max_abs_diff": graph_diff,
              "bond_logits_max_abs_diff": bond_diff, "export_reload_exact": True,
              "diffusion_backward_finite": True, "model_config": model.config.to_dict(),
              "best_epoch": saved.get("epoch"), "best_metrics": saved.get("metrics"),
              "exported_checkpoint_sha256": hashlib.sha256(exported.read_bytes()).hexdigest()}
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
