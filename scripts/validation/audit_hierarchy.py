"""Build and audit the topology-only near-fine/far-coarse hierarchy."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from coarse_gnn import HierarchyCache, HierarchyConfig, build_hierarchical_topology
from diffusion.bond_diffusion.data import JsonlGraphDataset, SmilesDataset


def graph_inputs(graph):
    pairs = torch.nonzero(torch.triu(graph.bonds > 0, diagonal=1), as_tuple=False)
    edges = pairs.T.contiguous()
    edge_labels = graph.bonds[pairs[:, 0], pairs[:, 1]] if pairs.numel() else torch.empty(0, dtype=torch.long)
    return edges.long(), graph.node_features.long(), edge_labels.long()


def permutation_stable(graph, topology, repeats, generator):
    edges, nodes, edge_labels = graph_inputs(graph)
    expected = topology.canonical_signature()
    for _ in range(repeats):
        permutation = torch.randperm(nodes.size(0), generator=generator)
        inverse = torch.argsort(permutation)
        moved = build_hierarchical_topology(
            nodes.size(0), inverse[edges], topology.config,
            node_labels=nodes[permutation], edge_labels=edge_labels,
        )
        if moved.canonical_signature() != expected:
            return False
    return True


def summarize(index, graph, topology, repeats, generator):
    diagnostics = topology.diagnostics()
    levels = []
    for component in topology.components:
        levels.append([
            {
                "tokens": level.num_tokens,
                "max_atoms": int(level.atom_counts.max()),
                "max_child_radius": int(level.child_radii.max()),
                "max_child_diameter": int(level.child_diameters.max()),
                "max_leaf_diameter": int(level.leaf_diameters.max()),
            }
            for level in component.levels
        ])
    return {
        "index": index,
        "smiles": graph.smiles,
        "atoms": graph.node_features.size(0),
        "physical_edges": int((graph.bonds > 0).triu(1).sum()),
        **diagnostics,
        "levels": levels,
        "permutation_stable": permutation_stable(graph, topology, repeats, generator),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Graph JSONL, SMILES CSV, .smi or .txt file")
    source.add_argument("--smiles", nargs="+", help="Explicit SMILES examples")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/hierarchy_audit.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hierarchy_cache"))
    args = parser.parse_args()
    if args.limit < 1 or args.permutations < 0:
        parser.error("--limit must be positive and --permutations nonnegative")
    if args.smiles is not None:
        dataset = SmilesDataset(args.smiles)
    elif args.input.suffix.lower() == ".jsonl":
        dataset = JsonlGraphDataset(args.input)
    else:
        dataset = SmilesDataset.from_file(args.input, args.smiles_column)

    config = HierarchyConfig()
    cache = HierarchyCache(args.cache_dir)
    generator = torch.Generator().manual_seed(20260918)
    records = []
    for index in range(min(args.limit, len(dataset))):
        graph = dataset[index]
        edges, nodes, edge_labels = graph_inputs(graph)
        topology = cache.get_or_build(
            nodes.size(0), edges, config,
            node_labels=nodes, edge_labels=edge_labels,
        )
        records.append(summarize(index, graph, topology, args.permutations, generator))
        print(
            f"{index + 1}/{min(args.limit, len(dataset))}: atoms={nodes.size(0)} "
            f"levels={records[-1]['level_counts']}",
            flush=True,
        )

    aggregate = {
        "graphs": len(records),
        "p_build_l2_or_l3": sum(record["built_l2"] for record in records) / max(1, len(records)),
        "p_build_l3": sum(record["built_l3"] for record in records) / max(1, len(records)),
        "p_use_l2_or_l3": sum(record["used_l2"] or record["used_l3"] for record in records) / max(1, len(records)),
        "p_use_l3": sum(record["used_l3"] for record in records) / max(1, len(records)),
        "mean_context_ratio": sum(record["mean_context_ratio"] for record in records) / max(1, len(records)),
        "permutation_stable": all(record["permutation_stable"] for record in records),
    }
    report = {
        "config": asdict(config),
        "cache": cache.stats,
        "aggregate": aggregate,
        "graphs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
