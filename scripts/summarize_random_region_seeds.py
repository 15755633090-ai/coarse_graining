"""Audit and summarize five downstream seeds sharing the seed-0 encoder."""
import hashlib
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    source_path = ROOT / "runs/lipo_formal_stage1_seed0_replay_v1/best.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    base = torch.load(ROOT / "runs/lipo_formal_stage2_seed0_base_replay_v1/best.pt", map_location="cpu", weights_only=False)
    rows, checks, reference = [], {}, None
    for seed in range(5):
        directory = ROOT / f"runs/lipo_random_region_seed{seed}_r2_optimized_v1"
        c = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row["valid_rmse"])
        if reference is None:
            reference = c
        checks[str(seed)] = {
            "completed": bool(c.get("test_metrics")),
            "history_matches": history == c["history"],
            "best_epoch_matches": best["epoch"] == c["best_epoch"] == c["model_state_epoch"],
            "seeds_match": c["model_seed"] == c["data_seed"] == seed and c["partition_seed"] == 100000,
            "mode_matches": c["experiment_mode"] == "random_region_stage2_frozen",
            "protocol_matches": c["training_protocol"] == reference["training_protocol"],
            "model_config_matches": c["model_config"] == reference["model_config"],
            "source_hash_matches": c["provenance"]["stage1_encoder_source"]["sha256"] == source_hash,
            "split_matches": c["provenance"]["dataset"] == source["provenance"]["dataset"],
            "scaler_matches": all(torch.equal(v, c["target_scaler"][k]) for k, v in source["target_scaler"].items()),
            "encoder_frozen_best_last": all(torch.equal(v, c["model_state"][k]) and torch.equal(v, last["model_state"][k]) for k, v in source["model_state"].items() if k.startswith("encoder.")),
        }
        if not c.get("test_metrics"):
            raise RuntimeError(f"seed {seed} has not completed test evaluation")
        rows.append(dict(seed=seed, epochs=len(history), best_epoch=best["epoch"]+1,
                         validation_rmse=best["valid_rmse"],
                         **{f"test_{k}": c["test_metrics"][k] for k in ("rmse", "mae", "r2")}))
    aggregate = {metric: {"mean": statistics.mean(r[metric] for r in rows),
                          "sample_std": statistics.stdev(r[metric] for r in rows),
                          "min": min(r[metric] for r in rows), "max": max(r[metric] for r in rows)}
                 for metric in ("validation_rmse", "test_rmse", "test_mae", "test_r2")}
    references = {name: {"seed": 0, "test_rmse": c["test_metrics"]["rmse"],
                         "new_seeds_below_reference": sum(r["test_rmse"] < c["test_metrics"]["rmse"] for r in rows)}
                  for name, c in (("Stage-1 Base", source), ("Stage-2 Base", base))}
    report = dict(rows=rows, aggregate=aggregate, reference_single_seed=references, checks=checks,
                  all_checks_pass=all(v for group in checks.values() for v in group.values()),
                  scope="Five downstream initialization/data-order seeds; shared fixed Stage-1 seed-0 encoder and fixed region sampling seed. Sample standard deviation uses ddof=1.")
    output = ROOT / "results/lipo_random_region_v1_5seeds"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Random Region v1：五个下游种子结果", "",
             "读取已有最佳 checkpoint，未重新评估测试集。所有种子共用 Stage-1 seed 0 encoder，区域采样 seed 固定为 100000。", "",
             "| Seed | 总轮数 | 最佳轮次 | 验证 RMSE | 测试 RMSE | 测试 MAE | 测试 R² |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['epochs']} | {r['best_epoch']} | {r['validation_rmse']:.5f} | {r['test_rmse']:.5f} | {r['test_mae']:.5f} | {r['test_r2']:.5f} |")
    lines += ["", "## 均值 ± 样本标准差（ddof=1）", ""]
    lines += [f"- {metric}: {v['mean']:.5f} ± {v['sample_std']:.5f}" for metric, v in aggregate.items()]
    lines += ["", "## 对照与限制", ""]
    lines += [f"- {name} 仅有 seed 0 参考：测试 RMSE {v['test_rmse']:.5f}；新模型 {v['new_seeds_below_reference']}/5 个种子低于该数值。" for name, v in references.items()]
    lines += ["- 以上不是五个配对 Base 的比较，不能据此断言统计显著提升。也未涵盖 encoder 微调种子或区域采样种子的变化。",
              "", "## 核验", ""]
    lines += [f"- Seed {seed}: {'PASS' if all(group.values()) else 'FAIL'}" for seed, group in checks.items()]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
