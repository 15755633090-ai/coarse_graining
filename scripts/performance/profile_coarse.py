"""Short real-Lipo throughput/profile probe; never writes formal training runs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import run_lipo_formal as formal
from coarse_gnn import TopologyCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/performance/before.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    legacy, settings, _, _ = formal.load_protocol(formal.PROJECT)
    source, splits, spec = legacy.build_property_data(settings.data_root, "lipo")
    indices = splits["train"][:32]
    started = time.perf_counter()
    rows = [source[i] for i in indices]
    data_seconds = time.perf_counter() - started
    cpu_batch = legacy.collate_property_batch(rows[:8])
    batch = cpu_batch.to("cuda")
    scaler = legacy.TargetScaler.fit(source.targets[splits["train"]], source.target_mask[splits["train"]])
    cache = TopologyCache(formal.PROJECT / "model/results_formal/05_coarse_gnn/_topology_cache", max_memory_entries=8192)
    # All these entries were created during formal preflight; no disk writes on hits.
    results = {}
    for variant in formal.VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        torch.use_deterministic_algorithms(True)
        model = formal.make_factory(legacy.create_property_model, cache)(variant, 1, settings.checkpoint, "cuda", 0.1)
        model.precompute_topologies(cpu_batch.graph)
        model.train()
        optimizer = torch.optim.AdamW([
            {"params": model.head.parameters(), "lr": 1e-3},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-4},
        ], weight_decay=settings.weight_decay)
        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with record_function("coarse_forward"):
                    prediction = model(batch.graph)
                loss = legacy.property_loss(prediction, batch.targets, batch.target_mask, spec, scaler, None)
            with record_function("coarse_backward"):
                loss.backward()
            with record_function("optimizer_step"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / 5
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        results[variant] = dict(seconds_per_micro_step=elapsed,
                                profiler_cpu_table=table,
                                cache=cache.stats)
        print(f"{variant}: {elapsed:.4f} seconds per 8-molecule step", flush=True)
        print(table, flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    formal.write_json(args.output, dict(data_parse_32_seconds=data_seconds,
                      node_counts=[len(row[0].node_features) for row in rows[:8]],
                      variants=results, note="Short warm-cache training probe, includes optimizer step; not a whole epoch."))


if __name__ == "__main__":
    main()
