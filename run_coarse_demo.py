"""Load the pretrained encoder and smoke-test the untrained downstream model."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from coarse_gnn import CoarseningConfig, NetworkConfig
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel
from diffusion.bond_diffusion.data import MoleculeGraph, collate_graphs, graph_from_smiles
from diffusion.bond_diffusion.trainer import load_encoder


def chain_graph(n: int) -> MoleculeGraph:
    bonds = torch.zeros(n, n, dtype=torch.long)
    index = torch.arange(n - 1)
    bonds[index, index + 1] = bonds[index + 1, index] = 1
    features = torch.zeros(n, 5, dtype=torch.long)
    features[:, 0] = 6
    features[:, 1] = 5
    features[:, 3] = 3
    features[:, 4] = (bonds > 0).sum(1)
    return MoleculeGraph(features, bonds, torch.zeros_like(bonds, dtype=torch.bool), f"synthetic_chain_{n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "diffusion/outputs/ogb_clean/encoder.pt")
    parser.add_argument("--smiles", nargs="+", help="Optional molecular inputs; default: synthetic chains of 1, 24, 100 nodes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.1)
    parser.add_argument("--max-residual-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--region-layers", type=int, default=2)
    parser.add_argument("--coarse-layers", type=int, default=3)
    parser.add_argument("--variant", choices=["enhanced", "base"], default="enhanced")
    parser.add_argument("--graph-pool", choices=["mean", "sum", "size_weighted_mean"], default=None)
    parser.add_argument("--no-region-edge-features", action="store_true")
    parser.add_argument("--no-coarse-edge-count", action="store_true")
    parser.add_argument("--no-coarse-edge-features", action="store_true")
    parser.add_argument("--no-size-feature", action="store_true")
    parser.add_argument("--legacy-coarsening", action="store_true", help="Disable canonicalization only for diagnostic comparisons")
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--backward", action="store_true", help="One optimizer step on synthetic targets to check gradients; not a scientific training run")
    parser.add_argument("--output", type=Path, default=Path("outputs/coarse_demo/canonical_report.json"))
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    encoder = load_encoder(args.checkpoint, args.device)
    base = args.variant == "base"
    network = NetworkConfig(
        input_dim=encoder.config.hidden_dim, hidden_dim=args.hidden_dim, edge_dim=4,
        region_layers=args.region_layers, coarse_layers=args.coarse_layers,
        graph_pool=args.graph_pool or ("mean" if base else "size_weighted_mean"),
        use_region_edge_features=not (base or args.no_region_edge_features),
        use_coarse_edge_count=not (base or args.no_coarse_edge_count),
        use_coarse_edge_features=not (base or args.no_coarse_edge_features),
        use_size_feature=not (base or args.no_size_feature),
    )
    coarsening = CoarseningConfig(args.radius, args.center_fraction, args.max_residual_size, canonicalize=not args.legacy_coarsening)
    from coarse_gnn import CoarseGraphPredictor
    model = DiffusionCoarseModel(encoder, CoarseGraphPredictor(network, coarsening), not args.finetune_encoder).to(args.device)
    graphs = [graph_from_smiles(s) for s in args.smiles] if args.smiles else [chain_graph(n) for n in (1, 24, 100)]
    batch = collate_graphs(graphs).to(args.device)
    training_check = None
    if args.backward:
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch).predictions
        targets = predictions.new_tensor([len(g.node_features) / 100 for g in graphs]).unsqueeze(1)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite smoke-test loss")
        loss.backward()
        gradient_parameters = [p for p in model.parameters() if p.grad is not None]
        if not gradient_parameters or not all(torch.isfinite(p.grad).all() for p in gradient_parameters):
            raise RuntimeError("Missing or nonfinite gradients")
        training_check = {"synthetic_loss": loss.item(), "parameters_with_gradients": len(gradient_parameters)}
        optimizer.step()
    model.eval()
    with torch.no_grad():
        result = model(batch)
    report = {
        "notice": "Framework smoke test only. Downstream weights are randomly initialized; predictions are not trained property estimates.",
        "checkpoint": str(args.checkpoint.resolve()), "device": args.device,
        "freeze_encoder": model.freeze_encoder,
        "network": asdict(network), "coarsening": asdict(coarsening),
        "prediction_shape": list(result.predictions.shape),
        "graph_embedding_shape": list(result.graph_embeddings.shape),
        "training_check": training_check,
        "graphs": [
            {
                "input": graph.smiles, "prediction": out.prediction.cpu().tolist(),
                "stats": out.topology.stats, "centers": out.topology.centers.tolist(),
                "owner": out.topology.owner.tolist(),
                "atom_order": out.topology.atom_order.tolist(),
                "owner_input": out.topology.owner_input.tolist(),
                "centers_input": out.topology.centers_input.tolist(),
                "canonical_signature": out.topology.canonical_signature(),
                "contexts": [c.tolist() for c in out.topology.contexts],
                "coarse_edges": out.topology.coarse_edges.T.tolist(),
                "coarse_edge_attr": out.coarse_edge_attr.cpu().tolist(),
            }
            for graph, out in zip(graphs, result.graphs)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report["notice"])
    for graph in report["graphs"]:
        print(graph["input"], json.dumps(graph["stats"]))
    print(f"Predictions: {tuple(result.predictions.shape)}; embeddings: {tuple(result.graph_embeddings.shape)}")
    if training_check:
        print("Backward/optimizer check:", training_check)
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
