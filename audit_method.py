"""Audit renumbering sensitivity and stereo information without training a model."""
import argparse
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch

from coarse_gnn import CoarseningConfig, NetworkConfig
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel
from diffusion.bond_diffusion.data import MoleculeGraph, collate_graphs, graph_from_smiles
from run_coarse_demo import chain_graph


def component_diameter(n, edges):
    adjacency = [[] for _ in range(n)]
    for u, v in edges.T.cpu().tolist():
        adjacency[u].append(v)
        adjacency[v].append(u)
    diameter = 0
    for start in range(n):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        diameter = max(diameter, max(distances.values()))
    return diameter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/method_audit/canonical.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", choices=["enhanced", "base"], default="enhanced")
    parser.add_argument("--legacy-coarsening", action="store_true", help="Reproduce input-ID-dependent coarsening; audit is expected to fail")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    if not 0 < args.tolerance < float("inf"):
        parser.error("--tolerance must be finite and positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    network = NetworkConfig.base(input_dim=128, edge_dim=0) if args.variant == "base" else None
    model = DiffusionCoarseModel.from_checkpoint(
        Path(__file__).resolve().parent / "diffusion/outputs/ogb_clean/encoder.pt",
        network=network, device=args.device,
        coarsening=CoarseningConfig(canonicalize=not args.legacy_coarsening),
    ).eval()
    samples = [
        ("chain_24", chain_graph(24)),
        ("chain_100", chain_graph(100)),
        ("PP_like_oligomer", graph_from_smiles("CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)C")),
    ]
    report = {
        "seed": args.seed, "permutations": args.permutations,
        "device": args.device, "tolerance": args.tolerance, "variant": args.variant,
        "network": asdict(model.predictor.config), "coarsening": asdict(model.predictor.coarsening),
        "notice": "Pretrained encoder, untrained downstream weights, eval mode. Every permutation reruns the encoder, canonicalization and complete coarsening. No persistent IDs are passed. This is a symmetry test, not trained prediction accuracy.",
        "graphs": [],
    }
    generator = torch.Generator().manual_seed(args.seed + 1)
    with torch.no_grad():
        for name, graph in samples:
            batch = collate_graphs([graph]).to(args.device)
            nodes = model.encoder.encode_nodes(batch.node_features, batch.bonds, batch.node_mask)[0]
            edges = torch.triu(graph.bonds > 0, diagonal=1).nonzero().T.contiguous()
            reference = model(batch).graphs[0]
            values, sizes, edge_counts, signatures, encoder_errors = [], [], [], [], []
            exact_edges = True
            for i in range(args.permutations):
                permutation = torch.randperm(len(nodes), generator=generator)
                moved = MoleculeGraph(graph.node_features[permutation], graph.bonds[permutation][:, permutation], graph.substructure_mask[permutation][:, permutation])
                moved_batch = collate_graphs([moved]).to(args.device)
                # Rerun the entire attributed-graph -> encoder -> coarse predictor.
                out = model(moved_batch).graphs[0]
                values.append(out.prediction.cpu())
                sizes.append(len(out.topology.cores))
                edge_counts.append(out.topology.coarse_edges.size(1))
                signatures.append(out.topology.canonical_signature())
                exact_edges &= torch.equal(reference.topology.coarse_edges, out.topology.coarse_edges)
                if i < 3:
                    encoded = model.encoder.encode_nodes(
                        moved_batch.node_features, moved_batch.bonds, moved_batch.node_mask,
                    )[0]
                    encoder_errors.append((encoded - nodes[permutation.to(nodes.device)]).abs().max().item())
            values = torch.stack(values).double()
            original_diameter = component_diameter(len(nodes), edges)
            coarse_diameter = component_diameter(len(reference.topology.cores), reference.topology.coarse_edges)
            item = {
                "name": name, "baseline_prediction": reference.prediction.tolist(),
                "prediction_std": values.std(dim=0, unbiased=False).tolist(),
                "prediction_range": (values.max(dim=0).values - values.min(dim=0).values).tolist(),
                "max_abs_change": (values - reference.prediction.cpu().double()).abs().max().item(),
                "coarse_node_counts_observed": sorted(set(sizes)),
                "coarse_edge_counts_observed": sorted(set(edge_counts)),
                "canonical_signature": reference.topology.canonical_signature(),
                "canonical_coarsening_identical": all(sig == reference.topology.canonical_signature() for sig in signatures),
                "coarse_edges_exactly_identical": exact_edges,
                "encoder_max_permutation_error": max(encoder_errors),
                "baseline_stats": reference.topology.stats,
                "original_diameter": original_diameter, "coarse_diameter": coarse_diameter,
                "diameter_ratio": coarse_diameter / original_diameter if original_diameter else None,
            }
            item["passed"] = (
                set(sizes) == {len(reference.topology.cores)}
                and set(edge_counts) == {reference.topology.coarse_edges.size(1)}
                and item["canonical_coarsening_identical"] and exact_edges
                and item["max_abs_change"] <= args.tolerance
            )
            report["graphs"].append(item)
            print(json.dumps(item), flush=True)
        stereo_pair = ["C[C@H](F)[C@H](Cl)Br", "C[C@H](F)[C@@H](Cl)Br"]
        first, second = map(graph_from_smiles, stereo_pair)
        report["stereo_probe"] = {
            "smiles": stereo_pair,
            "identical_node_features": torch.equal(first.node_features, second.node_features),
            "identical_bonds": torch.equal(first.bonds, second.bonds),
        }
    import igraph
    report["versions"] = {"torch": torch.__version__, "igraph": igraph.__version__}
    report["passed"] = all(item["passed"] for item in report["graphs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["stereo_probe"]), flush=True)
    if not report["passed"]:
        raise SystemExit("Permutation audit FAILED; see saved report")


if __name__ == "__main__":
    main()
