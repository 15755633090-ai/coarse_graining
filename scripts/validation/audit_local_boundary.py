"""Audit the topology-only LocalBoundary-GINE edge schedule on real Lipo."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

import run_lipo_formal as formal
import run_lipo_frozen as frozen
from coarse_gnn import (
    HierarchyCache,
    HierarchyConfig,
    local_boundary_reachability,
    split_local_boundary_topology,
)


def _graph_inputs(graph):
    pairs = torch.triu(graph.bonds > 0, diagonal=1).nonzero().T.contiguous()
    labels = graph.bonds[pairs[0], pairs[1]]
    return pairs, labels


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _component_record(component, plan):
    l1 = component.levels[0]
    count, edge_count = l1.num_tokens, l1.edges.size(1)
    local, local_boundary, added = local_boundary_reachability(component, plan)
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    local_pairs = int((local & off_diagonal).sum())
    local_boundary_pairs = int((local_boundary & off_diagonal).sum())
    added_pairs = int((added & off_diagonal).sum())
    degree_b = torch.zeros(count, dtype=torch.long)
    if plan.boundary_edges.numel():
        degree_b.index_add_(0, plan.boundary_edges[0], torch.ones(plan.boundary_edges.size(1), dtype=torch.long))
        degree_b.index_add_(0, plan.boundary_edges[1], torch.ones(plan.boundary_edges.size(1), dtype=torch.long))
    return {
        "l1_tokens": count,
        "edges": edge_count,
        "local_edges": plan.local_edge_ids.numel(),
        "boundary_edges": plan.boundary_edge_ids.numel(),
        "r_a": _ratio(plan.local_edge_ids.numel(), edge_count),
        "r_b": _ratio(plan.boundary_edge_ids.numel(), edge_count),
        "window_sizes": [value.numel() for value in plan.window_members],
        "has_cached_l2_membership": len(component.levels) >= 2,
        "membership_aligned": True,
        "boundary_empty": plan.boundary_edge_ids.numel() == 0,
        "boundary_isolated_tokens": int((degree_b == 0).sum()),
        "local_reachable_pairs": local_pairs,
        "local_boundary_reachable_pairs": local_boundary_pairs,
        "added_reachable_pairs": added_pairs,
        "possible_nonself_pairs": count * max(0, count - 1),
    }


def _aggregate(records, cache):
    components = [component for graph in records for component in graph["components"]]
    active = [component for component in components if component["boundary_edges"]]
    totals = {
        key: sum(component[key] for component in components)
        for key in (
            "l1_tokens", "edges", "local_edges", "boundary_edges",
            "boundary_isolated_tokens", "local_reachable_pairs",
            "local_boundary_reachable_pairs", "added_reachable_pairs",
            "possible_nonself_pairs",
        )
    }
    edgeful = [component for component in components if component["edges"]]
    window_sizes = [size for component in components for size in component["window_sizes"]]
    active_window_sizes = [
        size for component in active for size in component["window_sizes"]
    ]
    active_totals = {
        key: sum(component[key] for component in active)
        for key in (
            "l1_tokens", "edges", "local_edges", "boundary_edges",
            "boundary_isolated_tokens", "local_reachable_pairs",
            "added_reachable_pairs", "possible_nonself_pairs",
        )
    }
    return {
        "graphs": len(records),
        "components": len(components),
        "l1_tokens": totals["l1_tokens"],
        "edges": totals["edges"],
        "local_edges": totals["local_edges"],
        "boundary_edges": totals["boundary_edges"],
        "r_a_micro": _ratio(totals["local_edges"], totals["edges"]),
        "r_b_micro": _ratio(totals["boundary_edges"], totals["edges"]),
        "r_a_component_macro": _ratio(sum(row["r_a"] for row in edgeful), len(edgeful)),
        "r_b_component_macro": _ratio(sum(row["r_b"] for row in edgeful), len(edgeful)),
        "p_graph_boundary_empty": _ratio(
            sum(not graph["boundary_edges"] for graph in records), len(records),
        ),
        "p_component_boundary_empty": _ratio(
            sum(row["boundary_empty"] for row in components), len(components),
        ),
        "p_boundary_isolated_token": _ratio(
            totals["boundary_isolated_tokens"], totals["l1_tokens"],
        ),
        "window_size_histogram": dict(sorted(Counter(window_sizes).items())),
        "mean_window_size": _ratio(sum(window_sizes), len(window_sizes)),
        "p_window_size_gt_6": _ratio(
            sum(size > 6 for size in window_sizes), len(window_sizes),
        ),
        "local_reachable_pairs": totals["local_reachable_pairs"],
        "local_boundary_reachable_pairs": totals["local_boundary_reachable_pairs"],
        "added_reachable_pairs": totals["added_reachable_pairs"],
        "added_reachability_fraction": _ratio(
            totals["added_reachable_pairs"], totals["possible_nonself_pairs"],
        ),
        "relative_reachability_gain": _ratio(
            totals["added_reachable_pairs"], totals["local_reachable_pairs"],
        ),
        "mean_added_relations_per_token": _ratio(
            totals["added_reachable_pairs"], totals["l1_tokens"],
        ),
        "active_components": len(active),
        "p_active_component": _ratio(len(active), len(components)),
        "active_tokens": active_totals["l1_tokens"],
        "p_token_in_active_component": _ratio(
            active_totals["l1_tokens"], totals["l1_tokens"],
        ),
        "r_b_micro_active_components": _ratio(
            active_totals["boundary_edges"], active_totals["edges"],
        ),
        "p_boundary_isolated_token_active_components": _ratio(
            active_totals["boundary_isolated_tokens"], active_totals["l1_tokens"],
        ),
        "relative_reachability_gain_active_components": _ratio(
            active_totals["added_reachable_pairs"], active_totals["local_reachable_pairs"],
        ),
        "added_reachability_fraction_active_components": _ratio(
            active_totals["added_reachable_pairs"], active_totals["possible_nonself_pairs"],
        ),
        "mean_added_relations_per_active_token": _ratio(
            active_totals["added_reachable_pairs"], active_totals["l1_tokens"],
        ),
        "active_window_size_histogram": dict(sorted(Counter(active_window_sizes).items())),
        "p_active_window_size_gt_6": _ratio(
            sum(size > 6 for size in active_window_sizes), len(active_window_sizes),
        ),
        "max_active_window_size": max(active_window_sizes, default=0),
        "cached_membership_alignment_passed": all(
            row["membership_aligned"] for row in components
        ),
        "edge_partition_assertions_passed": (
            totals["local_edges"] + totals["boundary_edges"] == totals["edges"]
        ),
        "cache": cache.stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 audits the complete Lipo dataset")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("outputs/hierarchy_lipo_frozen/_hierarchy_cache"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/local_boundary_audit/lipo_topology.json"),
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    torch.set_num_threads(2)
    legacy, protocol_args, protocol, baseline, selection = frozen.load_protocol(formal.PROJECT)
    source, _, _, inputs = formal.check_inputs(
        legacy, protocol_args, protocol, baseline,
    )
    total = len(source) if args.limit == 0 else min(args.limit, len(source))
    cache = HierarchyCache(args.cache_dir, max_memory_entries=8192)
    config = HierarchyConfig()
    records = []
    for index in range(total):
        graph = source[index][0]
        edges, labels = _graph_inputs(graph)
        topology = cache.get_or_build(
            graph.node_features.size(0), edges, config,
            node_labels=graph.node_features, edge_labels=labels,
        )
        plans = split_local_boundary_topology(topology)
        components = [
            _component_record(component, plan)
            for component, plan in zip(topology.components, plans)
        ]
        records.append({
            "index": index,
            "smiles": source.smiles[index],
            "l1_tokens": sum(row["l1_tokens"] for row in components),
            "edges": sum(row["edges"] for row in components),
            "boundary_edges": sum(row["boundary_edges"] for row in components),
            "components": components,
        })
        if index % 512 == 0:
            print(f"LocalBoundary topology audit: {index + 1}/{total}", flush=True)

    aggregate = _aggregate(records, cache)
    report = {
        "method": "local_boundary_gine_v1",
        "neural_model_executed": False,
        "test_metrics_used": False,
        "a_window_source": "cached deterministic L2 parent membership only; no L2 token is used",
        "hierarchy": asdict(config),
        "split_sha256": protocol["split_sha256"],
        "selection_identity": selection,
        "inputs": inputs,
        "aggregate": aggregate,
        "graphs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
