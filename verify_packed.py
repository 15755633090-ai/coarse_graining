"""Real-device equivalence/RNG and throughput audit, no formal result writes."""
import argparse
import time
import math
from pathlib import Path

import torch

import run_lipo_formal as formal
from coarse_gnn import TopologyCache
from coarse_gnn.packed import packed_predict, prepare_graph_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/performance/packed_audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    legacy, settings, _, _ = formal.load_protocol(formal.PROJECT)
    source, splits, spec = legacy.build_property_data(settings.data_root, "lipo")
    # Two nonidentical micro-batches, official train samples only.
    rows = [source[i] for i in splits["train"][:16]]
    cpu_batches = [legacy.collate_property_batch(rows[i:i + 8]) for i in (0, 8)]
    cache = TopologyCache(formal.PROJECT / "model/results_formal/05_coarse_gnn/_topology_cache")
    results = []
    passed = True
    for variant in formal.VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        torch.use_deterministic_algorithms(True)
        model = formal.make_factory(legacy.create_property_model, cache)(variant, 1, settings.checkpoint, "cuda", 0.1).train()
        resume = formal.PROJECT / "model/results_formal/05_coarse_gnn/lipo" / variant / "tuning/seed_42/trial_0/resume.pt"
        resume_checked = False
        if resume.exists():
            checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer = torch.optim.AdamW([
                {"params": model.head.parameters(), "lr": 1e-3},
                {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-4},
            ])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            resume_checked = True
            del optimizer, checkpoint
        for number, cpu in enumerate(cpu_batches):
            raw = cpu.to("cuda")
            prepared = prepare_graph_batch(cpu.graph, model.predictor).to("cuda")
            for amp in (False, True):
                states, predictions, gradients = [], [], []
                for packed in (False, True):
                    model.zero_grad(set_to_none=True)
                    legacy.seed_everything(123, deterministic=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        if packed:
                            nodes = model.encoder.encode_nodes(prepared.node_features, prepared.bonds, prepared.node_mask)
                            output = packed_predict(model.predictor, nodes, prepared)
                        else:
                            output = model(raw.graph)
                        loss = output.float().square().mean()
                    states.append(torch.cuda.get_rng_state().clone())
                    predictions.append(output.detach().float().cpu())
                    loss.backward()
                    gradients.append({name: p.grad.detach().float().cpu().clone()
                                      for name, p in model.named_parameters() if p.grad is not None})
                max_grad = max((gradients[0][name] - gradients[1][name]).abs().max().item() for name in gradients[0])
                normalized_grad = max((gradients[0][name] - gradients[1][name]).abs().max().item() /
                                      max(gradients[0][name].abs().max().item(), 1e-6) for name in gradients[0])
                gradient_relative_l2 = math.sqrt(sum((gradients[0][name] - gradients[1][name]).double().square().sum().item() for name in gradients[0])) / max(math.sqrt(sum(value.double().square().sum().item() for value in gradients[0].values())), 1e-12)
                row = dict(variant=variant, micro_batch=number, amp=amp,
                           prediction_max_abs=(predictions[0] - predictions[1]).abs().max().item(),
                           gradient_max_abs=max_grad, gradient_max_relative_to_tensor_max=normalized_grad,
                           gradient_relative_l2=gradient_relative_l2, real_resume_state_loaded=resume_checked,
                           rng_identical=torch.equal(states[0], states[1]))
                row["passed"] = (row["rng_identical"] and row["prediction_max_abs"] <= (0.01 if amp else 2e-5)
                                 and gradient_relative_l2 <= (0.02 if amp else 1e-4))
                passed &= row["passed"]
                results.append(row)
                print(row, flush=True)
        # Both paths timed on identical data and model weights, with dropout enabled.
        seconds = {}
        for packed in (False, True):
            def step():
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if packed:
                        nodes = model.encoder.encode_nodes(prepared.node_features, prepared.bonds, prepared.node_mask)
                        out = packed_predict(model.predictor, nodes, prepared)
                    else:
                        out = model(raw.graph)
                    loss = out.float().square().mean()
                loss.backward()
            for _ in range(2):
                step()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(8):
                step()
            torch.cuda.synchronize()
            seconds["packed" if packed else "serial"] = (time.perf_counter() - started) / 8
        results.append(dict(variant=variant, seconds=seconds, speedup=seconds["serial"] / seconds["packed"]))
        print(results[-1], flush=True)
    formal.write_json(args.output, dict(results=results, passed=passed,
                      source_hashes={key.replace("\\", "/"): value["sha256"] for key, value in formal.source_identity(legacy).items()},
                      note="Same weights, inputs and seed; numerical differences explicitly reported, no formal training performed."))
    if not passed:
        raise SystemExit("Packed numerical/RNG audit failed")


if __name__ == "__main__":
    main()
