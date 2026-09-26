# -*- coding: utf-8 -*-
"""Summarize saved replay-v1 results without reevaluating the test split."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ["lipo_formal_stage1_seed0_replay_v1", "lipo_formal_stage2_seed0_base_replay_v1", "lipo_formal_stage2_seed0_multiscale_replay_v1"]
    labels = ["Stage-1 Base", "Stage-2 Base", "Stage-2 Multiscale"]
    paths = [ROOT / "runs" / name / "best.pt" for name in names]
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    source, base, coarse = checkpoints
    source_hash = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    checks = {}
    rows = []
    for label, path, c in zip(labels, paths, checkpoints):
        best = min(c["history"], key=lambda r: r["valid_rmse"])
        checks[label + " best_epoch_matches_history"] = best["epoch"] == c["best_epoch"] == c["model_state_epoch"]
        checks[label + " completed_with_test"] = bool(c.get("test_metrics")) and c["epochs_without_improvement"] == 30
        rows.append(dict(experiment=label, run=path.parent.name, epochs=len(c["history"]), best_epoch_one_based=c["best_epoch"]+1, validation_rmse=best["valid_rmse"], test_rmse=c["test_metrics"]["rmse"], test_mae=c["test_metrics"]["mae"], test_r2=c["test_metrics"]["r2"]))
        if c is not source:
            last = torch.load(path.with_name("last.pt"), map_location="cpu", weights_only=False)
            checks[label + " source_hash_matches"] = c["provenance"]["stage1_encoder_source"]["sha256"] == source_hash
            checks[label + " encoder_best_and_last_unchanged"] = all(torch.equal(v, c["model_state"][k]) and torch.equal(v, last["last_model_state"][k]) for k,v in source["model_state"].items() if k.startswith("encoder."))
            checks[label + " scaler_matches_source"] = all(torch.equal(v,source["target_scaler"][k]) for k,v in c["target_scaler"].items())
    checks["paired_training_protocol_equal"] = base["training_protocol"] == coarse["training_protocol"]
    checks["paired_seeds_equal"] = all(base[k] == coarse[k] for k in ("model_seed", "data_seed", "partition_seed"))
    checks["dataset_and_split_equal"] = all(c["provenance"]["dataset"] == source["provenance"]["dataset"] for c in checkpoints)
    old = json.loads((ROOT.parent / "model/results_formal/04_diffusion/lipo/pretrained_finetune/seed_0/history.json").read_text())
    checks["stage1_all_validation_epochs_reproduce_reference"] = len(old)==len(source["history"]) and all(a["valid_metrics"]["rmse"]==b["valid_rmse"] for a,b in zip(old,source["history"]))
    delta = rows[2]["test_rmse"]-rows[1]["test_rmse"]
    differences = {"multiscale_minus_base_test_rmse": delta, "multiscale_relative_rmse_increase_percent": 100*delta/rows[1]["test_rmse"], "multiscale_minus_base_validation_rmse": rows[2]["validation_rmse"]-rows[1]["validation_rmse"], "base2_minus_stage1_test_rmse": rows[1]["test_rmse"]-rows[0]["test_rmse"]}
    out = ROOT / "results/lipo_seed0_replay_v1"
    out.mkdir(exist_ok=True)
    report = dict(dataset="lipo", split="scaffold", seed=0, rows=rows, comparisons=differences, checks=checks, paired_protocol=base["training_protocol"], stage1_source_sha256=source_hash)
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Lipo seed 0：修正执行协议后的结果", "", "仅汇总保存的最佳 checkpoint，不重新评估测试集。轮次从 1 开始。", "", "| 实验 | 总轮数 | 最佳轮次 | 验证 RMSE ↓ | 测试 RMSE ↓ | 测试 MAE ↓ | 测试 R² ↑ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['epochs']} | {r['best_epoch_one_based']} | {r['validation_rmse']:.5f} | {r['test_rmse']:.5f} | {r['test_mae']:.5f} | {r['test_r2']:.5f} |")
    lines += ["", f"粗粒化相对 Base2：测试 RMSE 增加 {delta:.5f}（{differences['multiscale_relative_rmse_increase_percent']:.2f}%），验证 RMSE 增加 {differences['multiscale_minus_base_validation_rmse']:.5f}。", "", "本次 seed 0 下，当前粗粒化实现未带来性能收益。该结果不构成跨随机种子的统计结论，也不能单凭指标判定具体退化机制。", "", "Stage-1 的全部 107 轮验证轨迹复现旧 Base。两组 Stage-2 使用相同来源 encoder、数据划分、scaler、随机种子和训练超参数；best/last encoder 权重与来源逐元素一致。两组读出结构不同，因此结果比较的是完整粗粒化方案与 Base2，而非隔离单一模块的因果效应。", "", "## 核对", ""]
    lines += [f"- {'PASS' if v else 'FAIL'}: {k}" for k,v in checks.items()]
    (out / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"rows":rows,"comparisons":differences,"all_checks_pass":all(checks.values()),"output":str(out)},indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
