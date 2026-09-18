"""Benchmark real CUDA packed hierarchy throughput with random vs bucket batches."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from coarse_gnn import (
    AtomCountBucketBatchSampler,
    HierarchicalPredictor,
    HierarchyNetworkConfig,
    build_hierarchical_topology,
    pack_hierarchy,
)
from coarse_gnn.diffusion_adapter import encode_nodes_with_intermediates
from diffusion.bond_diffusion.data import SmilesDataset, collate_graphs
from diffusion.bond_diffusion.model import masked_mean
from diffusion.bond_diffusion.trainer import load_encoder


def topology(graph):
    edges = torch.triu(graph.bonds > 0, diagonal=1).nonzero().T.contiguous()
    labels = graph.bonds[edges[0], edges[1]]
    return build_hierarchical_topology(
        graph.node_features.size(0), edges,
        node_labels=graph.node_features, edge_labels=labels,
    )


def random_batches(count, batch_size, seed):
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


def prepare_batches(graphs, topologies, index_batches):
    prepared = []
    for indices in index_batches:
        selected = [graphs[index] for index in indices]
        batch = collate_graphs(selected)
        plan = pack_hierarchy(
            [topologies[index] for index in indices], batch.node_features.size(1),
        )
        prepared.append((batch, plan, [graph.node_features.size(0) for graph in selected]))
    return prepared


def benchmark_interleaved(
    encoder, predictor, base_head, prepared_modes, device, warmup_passes, repeats,
):
    parameters = (*encoder.parameters(), *predictor.parameters(), *base_head.parameters())

    def clear_gradients():
        for parameter in parameters:
            parameter.grad = None

    def step(batch_cpu, plan_cpu):
        batch = batch_cpu.to(device)
        plan = plan_cpu.to(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final, captured = encode_nodes_with_intermediates(
                encoder, batch.node_features, batch.bonds, batch.node_mask, layers=(2,),
            )
            base_embedding = masked_mean(final, batch.node_mask, dim=1)
            base_prediction = base_head(base_embedding)
            output = predictor(captured[2], base_embedding, base_prediction, plan)
            loss = output.prediction.float().square().mean()
        loss.backward()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop)

    modes = tuple(prepared_modes)
    steps = len(prepared_modes[modes[0]])
    if any(len(prepared_modes[mode]) != steps for mode in modes):
        raise ValueError("interleaved modes must have the same number of batches")
    for _ in range(warmup_passes):
        for index in range(steps):
            for mode in modes if index % 2 == 0 else reversed(modes):
                batch, plan, _ = prepared_modes[mode][index]
                step(batch, plan)
                clear_gradients()

    records = {
        mode: {"elapsed": [], "samples": 0, "padding": [], "peak": 0.0}
        for mode in modes
    }
    sampler = subprocess.Popen(
        [
            "nvidia-smi", "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits", "--loop-ms=100",
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        for repeat in range(repeats):
            for index in range(steps):
                order = modes if (repeat + index) % 2 == 0 else tuple(reversed(modes))
                for mode in order:
                    batch, plan, sizes = prepared_modes[mode][index]
                    torch.cuda.reset_peak_memory_stats(device)
                    records[mode]["elapsed"].append(step(batch, plan))
                    records[mode]["samples"] += len(sizes)
                    records[mode]["padding"].append(max(sizes) / (sum(sizes) / len(sizes)))
                    records[mode]["peak"] = max(
                        records[mode]["peak"], torch.cuda.max_memory_allocated(device) / 2**20,
                    )
                    clear_gradients()
    finally:
        sampler.terminate()
        output, _ = sampler.communicate(timeout=5)
    utilization = []
    for line in output.splitlines():
        try:
            utilization.append(float(line.strip().rstrip("%")))
        except ValueError:
            pass
    shared_utilization = sum(utilization) / len(utilization) if utilization else None
    results = {}
    for mode, record in records.items():
        seconds = sum(record["elapsed"]) / 1000
        results[mode] = {
            "samples": record["samples"],
            "steps": len(record["elapsed"]),
            "samples_per_second": record["samples"] / seconds,
            "mean_forward_backward_ms": sum(record["elapsed"]) / len(record["elapsed"]),
            "peak_memory_mib": record["peak"],
            "mean_gpu_utilization_percent_shared_interleaved": shared_utilization,
            "mean_padding_max_over_mean": sum(record["padding"]) / len(record["padding"]),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="diffusion/datasets/ogb_pretrain_clean.csv")
    parser.add_argument("--checkpoint", default="diffusion/outputs/ogb_clean/encoder.pt")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--output", default="outputs/hierarchy_gpu_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    random.seed(args.seed)
    dataset = SmilesDataset.from_file(args.data)
    selected = random.sample(range(len(dataset)), args.samples)
    graphs = [dataset[index] for index in selected]
    started = time.perf_counter()
    topologies = [topology(graph) for graph in graphs]
    preprocessing_seconds = time.perf_counter() - started
    counts = [graph.node_features.size(0) for graph in graphs]

    device = torch.device("cuda")
    encoder = load_encoder(args.checkpoint, device).eval()
    predictor = HierarchicalPredictor(HierarchyNetworkConfig(
        input_dim=encoder.config.hidden_dim,
        base_dim=encoder.config.hidden_dim,
        hidden_dim=128,
        dropout=0,
    )).to(device).eval()
    base_head = nn.Linear(encoder.config.hidden_dim, 1).to(device).eval()
    results = {
        "device": torch.cuda.get_device_name(device),
        "checkpoint": args.checkpoint,
        "samples": args.samples,
        "atom_count": {
            "min": min(counts), "mean": sum(counts) / len(counts), "max": max(counts),
        },
        "offline_topology_seconds": preprocessing_seconds,
        "configurations": {},
    }
    for batch_size in args.batch_sizes:
        modes = {
            "random": random_batches(len(graphs), batch_size, args.seed),
            "bucket": list(AtomCountBucketBatchSampler(
                counts, batch_size, seed=args.seed, bucket_size_multiplier=20,
            )),
        }
        prepared_modes = {
            mode: prepare_batches(graphs, topologies, batches)
            for mode, batches in modes.items()
        }
        measured = benchmark_interleaved(
            encoder, predictor, base_head, prepared_modes, device,
            args.warmup_passes, args.repeats,
        )
        for mode, values in measured.items():
            results["configurations"][f"batch_{batch_size}_{mode}"] = values
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
