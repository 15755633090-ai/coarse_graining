"""Summarize completed seed-0 experiments from saved checkpoints only."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = {
        "Stage-1 Base": "lipo_formal_stage1_seed0_replay_v1",
        "Stage-2 Base": "lipo_formal_stage2_seed0_base_replay_v1",
        "旧多尺度模型": "lipo_formal_stage2_seed0_multiscale_replay_v1",
        "随机区域（原版）": "lipo_random_region_seed0_r2_v1",
        "随机区域（速度优化版）": "lipo_random_region_seed0_r2_optimized_v1",
    }
    checkpoints, rows, checks = [], [], {}
    for label, run in runs.items():
        directory = ROOT / "runs" / run
        c = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        checkpoints.append(c)
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        best = min(history, key=lambda r: r["valid_rmse"])
        checks[label + " history_matches_checkpoint"] = history == c["history"]
        checks[label + " selected_best_epoch"] = best["epoch"] == c["best_epoch"] == c["model_state_epoch"]
        checks[label + " completed_with_test"] = bool(c.get("test_metrics")) and c["epochs_without_improvement"] == 30
        metrics = c["test_metrics"]
        rows.append(dict(experiment=label, run=run, epochs=len(history), best_epoch=best["epoch"] + 1,
                         validation_rmse=best["valid_rmse"], test_rmse=metrics["rmse"],
                         test_mae=metrics["mae"], test_r2=metrics["r2"]))
    source, base, _, original, optimized = checkpoints
    source_hash = hashlib.sha256((ROOT / "runs" / runs["Stage-1 Base"] / "best.pt").read_bytes()).hexdigest()
    for label, c in zip(runs, checkpoints):
        checks[label + " dataset_split_matches"] = c["provenance"]["dataset"] == source["provenance"]["dataset"]
        if c is source:
            continue
        checks[label + " encoder_source_hash"] = c["provenance"]["stage1_encoder_source"]["sha256"] == source_hash
        last = torch.load(ROOT / "runs" / runs[label] / "last.pt", map_location="cpu", weights_only=False)
        checks[label + " frozen_encoder_best_last"] = all(
            torch.equal(value, c["model_state"][key]) and torch.equal(value, last["model_state"][key])
            for key, value in source["model_state"].items() if key.startswith("encoder.")
        )
        checks[label + " target_scaler_matches"] = all(torch.equal(value, c["target_scaler"][key]) for key, value in source["target_scaler"].items())
    checks["optimization_full_history_exact"] = original["history"] == optimized["history"]
    checks["optimization_test_metrics_exact"] = original["test_metrics"] == optimized["test_metrics"]
    checks["optimization_model_config_exact"] = original["model_config"] == optimized["model_config"]
    checks["optimization_training_protocol_exact"] = original["training_protocol"] == optimized["training_protocol"]
    checks["optimization_best_weights_exact"] = original["model_state"].keys() == optimized["model_state"].keys() and all(
        torch.equal(v, optimized["model_state"][k]) for k, v in original["model_state"].items())
    checks["paired_base_common_training_protocol"] = base["training_protocol"] == {
        k: v for k, v in optimized["training_protocol"].items() if k != "random_regions"}
    checks["paired_seeds"] = all(base[k] == original[k] == optimized[k] for k in ("model_seed", "data_seed", "partition_seed"))
    comparisons = {}
    for row in rows[:2]:
        improvement = row["test_rmse"] - rows[-1]["test_rmse"]
        comparisons[row["experiment"]] = {
            "test_rmse_reduction": improvement,
            "test_rmse_reduction_percent": 100 * improvement / row["test_rmse"],
            "test_mae_change": rows[-1]["test_mae"] - row["test_mae"],
        }
    report = dict(dataset="lipo", split="scaffold", seed=0, rows=rows, comparisons=comparisons,
                  checks=checks, random_region_config=optimized["training_protocol"]["random_regions"],
                  all_checks_pass=all(checks.values()))
    out = ROOT / "results/lipo_random_region_seed0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Lipo seed 0：随机区域模型结果汇总", "",
             "从保存的 checkpoint 和训练日志汇总，未重新运行测试集。轮次从 1 开始。", "",
             "| 模型 | 总轮数 | 最佳轮次 | 验证 RMSE ↓ | 测试 RMSE ↓ | 测试 MAE ↓ | 测试 R² ↑ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['epochs']} | {r['best_epoch']} | {r['validation_rmse']:.5f} | {r['test_rmse']:.5f} | {r['test_mae']:.5f} | {r['test_r2']:.5f} |")
    lines += ["", "## 对比", ""]
    for label, values in comparisons.items():
        lines.append(f"- 相比 {label}，测试 RMSE 下降 {values['test_rmse_reduction']:.5f}（{values['test_rmse_reduction_percent']:.2f}%）；测试 MAE 变化为 {values['test_mae_change']:+.5f}。")
    lines += ["", "## 解释与边界", "",
              "- 新模型保留 H4 Sum/Mean 表示，与随机区域表示拼接后重新训练预测头；未使用冻结 Base predictor 加 residual correction。",
              "- 配置：r=2，s=8，Kmax=8，单层四头距离 bias attention，训练 1 个视角、验证和测试 5 个固定视角。",
              "- Stage-2 Base 与新模型使用同一 Stage-1 来源的冻结 encoder；Stage-1 Base 为微调阶段的较强参考。",
              "- 优化版与原版是同一 seed 的实现复核，不能当作两个独立随机种子。当前结果不代表多 seed 稳定性。",
              "- 训练日志没有可靠的逐轮耗时记录，本次汇总不推算整轮加速比例。",
              "", "## 核验", ""]
    lines += [f"- {'PASS' if value else 'FAIL'}: {key}" for key, value in checks.items()]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
