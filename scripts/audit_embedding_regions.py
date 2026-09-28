"""Small Lipo train-only smoke audit; never evaluate test or start a full run."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from diffusion_encoder.data import OGBMoleculePropertyDataset, collate_property_records
from multiscale_tokenizer.training import (
    TASK_SPECS, build_property_model, load_property_model,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = ROOT / "runs/lipo_formal_stage1_seed0_replay_v1/best.pt"
    model = build_property_model(
        spec=TASK_SPECS["lipo"], experiment_mode="embedding_region_stage2_frozen",
        model_seed=0, partition_seed=100000, dropout=0.1, device=device,
        encoder_init_checkpoint=source, eval_views=1,
    )
    dataset = OGBMoleculePropertyDataset(ROOT / "datasets/lipo", split="train")
    batch = collate_property_records([dataset[i] for i in range(8)]).to(device)
    graph = batch.graph
    args = graph.node_features, graph.bonds, graph.node_mask
    model.prepare_embedding_regions(*args)
    initial_encoder = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    with torch.autocast(device.type, dtype=torch.bfloat16):
        output = model(*args)
        loss = output.prediction.float().square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    optimizer.step()
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in initial_encoder.items())
    model.eval()
    with torch.no_grad():
        expected = model(*args).prediction
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "smoke.pt"
        torch.save({"schema_version": 4, "model_config": asdict(model.config),
                    "model_state": model.state_dict(), "task_spec": asdict(TASK_SPECS["lipo"]),
                    "embedding_region_cache": model.embedding_region_cache}, checkpoint)
        restored, _, _ = load_property_model(checkpoint, device=device)
        with torch.no_grad():
            actual = restored(*args).prediction
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert len(restored.embedding_region_cache) == len(model.embedding_region_cache)
    result = {"device": str(device), "source": str(source), "train_molecules": 8,
              "token_counts": output.token_counts[:, 3].tolist(),
              "finite_forward_backward": bool(torch.isfinite(loss)),
              "encoder_unchanged": True, "checkpoint_roundtrip_exact": True,
              "test_evaluated": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
