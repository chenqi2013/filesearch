"""Combine frozen comparison outputs without rerunning or adjusting retrieval."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".codex-tmp/rwkv-qwen-comparison-20261029"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("comparison", root / "scripts/compare-rwkv-qwen.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    output = args.output.resolve()
    dataset = comparison.read(output / "frozen-dataset.json")
    reports = [comparison.read(output / path) for path in ["result.json", "rwkv-fp32-result.json", "qwen-fp32-result.json"]]
    expected = comparison.digest(output / "frozen-dataset.json")
    assert all(report["dataset_sha256"] == expected for report in reports)
    models = {key: value for report in reports for key, value in report["results"].items()}
    totals = {}
    for name, value in models.items():
        totals[name] = {corpus: comparison.summarize([row for row in value["rows"] if row["corpus"] == corpus], value["calibration"]["threshold"])
                        for corpus in ["real", "synthetic"]}
    vector_parity = {}
    for model in ["rwkv", "qwen"]:
        vector_parity[model] = {}
        for label in ["real", "synthetic", "raw", "instructed"]:
            a = np.load(output / f"{model}-{label}.npz")["vectors"]
            b = np.load(output / f"{model}-fp32-{label}.npz")["vectors"]
            scores = np.sum(a * b, axis=1)
            vector_parity[model][label] = dict(min=float(scores.min()), median=float(np.median(scores)), mean=float(scores.mean()))
    # Query-level paired bootstrap is a descriptive uncertainty check, not a new blind experiment.
    paired = {}
    for corpus in ["real", "synthetic"]:
        a = {(row["split"], row["query"]): row for row in models["rwkv_fp32_instructed"]["rows"] if row["corpus"] == corpus and row["kind"] == "positive"}
        b = {(row["split"], row["query"]): row for row in models["qwen_fp32_instructed"]["rows"] if row["corpus"] == corpus and row["kind"] == "positive"}
        assert a.keys() == b.keys()
        delta = np.array([int(b[key]["target_rank"] == 1) - int(a[key]["target_rank"] == 1) for key in a])
        rng = np.random.default_rng(20261029)
        means = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        paired[corpus] = dict(qwen_top1_advantage_pp=float(delta.mean() * 100),
                             descriptive_query_bootstrap_95_pp=(np.percentile(means, [2.5, 97.5]) * 100).tolist(),
                             qwen_only=int((delta == 1).sum()), rwkv_only=int((delta == -1).sum()),
                             note="Queries sharing documents are correlated; not a population confidence claim")
    verifier_spec = importlib.util.spec_from_file_location("ranking", root / "scripts/verify-search-ranking.py")
    verifier = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)
    assert verifier.vector_state(Path(dataset["source_database"])) == dataset["source_state"]
    result = dict(dataset_sha256=expected, source_unchanged=True, totals=totals, vector_parity=vector_parity,
                  paired=paired, models=models)
    comparison.save(output / "combined-result.json", result)
    columns = ["rwkv_instructed", "rwkv_fp32_instructed", "qwen_instructed", "qwen_fp32_instructed"]
    indexed = {name: {(row["corpus"], row["split"], row["query"]): row for row in models[name]["rows"]} for name in columns}
    lines = ["# RWKV / Qwen 逐条查询结果（2026-10-29）", "", "目标排名来自相同正文上的原始向量排序，未施加无答案过滤。均使用通用检索指令。", "",
             "RWKV 部署版为当前混合 INT8；RWKV 参考为官方 C FP32；Qwen 部署为本地 INT8 ONNX；Qwen 参考为官方权重 CPU FP32。", ""]
    for corpus, title in [("real", "现有 13 份资料：100 条有目标查询"), ("synthetic", "独立构造的 36 份文本：58 条有目标查询")]:
        lines += ["## " + title, "", "| 查询 | 目标文件 | RWKV 部署 | RWKV 参考 | Qwen INT8 | Qwen 参考 |", "|---|---|---:|---:|---:|---:|"]
        for case in dataset["cases"]:
            if case["corpus"] != corpus or case["kind"] != "positive":
                continue
            key = (corpus, case["split"], case["query"])
            ranks = [str(indexed[name][key]["target_rank"]) for name in columns]
            lines.append("| " + " | ".join([case["query"].replace("|", "\\|"), " / ".join(case["targets"]), *ranks]) + " |")
        lines.append("")
    lines += ["## 38 条无直接答案查询", "", "纯向量排序总会返回候选，不能把该输出当作拒答结果。下表展示最高相似度；不同模型分数不可共用同一阈值。校准和留出集结果见总报告。", "",
              "| 范围 | 查询 | RWKV 部署 | RWKV 参考 | Qwen INT8 | Qwen 参考 |", "|---|---|---:|---:|---:|---:|"]
    for case in dataset["cases"]:
        if case["kind"] != "negative":
            continue
        key = (case["corpus"], case["split"], case["query"])
        scores = [f'{indexed[name][key]["best_score"]:.4f}' for name in columns]
        lines.append("| " + " | ".join([case["corpus"] + "/" + case["split"], case["query"], *scores]) + " |")
    (root / "docs/rwkv-qwen-cases-20261029.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(totals=totals, vector_parity=vector_parity, paired=paired), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
