from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as parquet
import requests


REVISION = "7d52a069e0b37d976b3ed3f674a6180436c27574"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = (f"https://raw.githubusercontent.com/UKPLab/sentence-transformers/{REVISION}/"
           "sentence_transformers/evaluation/InformationRetrievalEvaluator.py")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    source = response.text
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "InformationRetrievalEvaluator")
    methods = [node for node in definition.body if isinstance(node, ast.FunctionDef)
               and node.name in ["compute_metrics", "compute_dcg_at_k"]]
    for method in methods:
        method.decorator_list = []
    namespace = {"np": np}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<sentence-transformers>", "exec"), namespace)
    recorded = json.loads(args.results.read_text(encoding="utf-8"))
    vector_root = args.results.parent / (args.results.stem + "-vectors")
    comparisons = []
    for entry in recorded["results"]:
        name = entry["name"]
        filename = f"Nano{name}-00000-of-00001.parquet"
        root = args.data / recorded["dataset_revision"]
        rows = {kind: parquet.read_table(root / kind / filename).to_pylist()
                for kind in ["corpus", "queries", "qrels"]}
        vectors = np.load(vector_root / f"{name}.npz")
        scores = vectors["queries"] @ vectors["documents"].T
        ranking = np.argsort(-scores, axis=1, kind="stable")[:, :100]
        hits = [[{"corpus_id": rows["corpus"][index]["_id"], "score": float(scores[position, index])}
                 for index in indexes] for position, indexes in enumerate(ranking)]
        relevant = {}
        for row in rows["qrels"]:
            if row.get("score", 1) > 0:
                relevant.setdefault(row["query-id"], set()).add(row["corpus-id"])
        evaluator = SimpleNamespace(accuracy_at_k=[1, 10], precision_recall_at_k=[10],
                                    mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[100],
                                    queries_ids=[row["_id"] for row in rows["queries"]],
                                    queries=rows["queries"], relevant_docs=relevant,
                                    compute_dcg_at_k=namespace["compute_dcg_at_k"])
        official = namespace["compute_metrics"](evaluator, hits)
        differences = {
            "ndcg@10": abs(float(official["ndcg@k"][10]) - entry["ndcg@10"]),
            "recall@10": abs(float(official["recall@k"][10]) - entry["recall@10"]),
            "accuracy@1": abs(float(official["accuracy@k"][1]) - entry["accuracy@1"]),
            "hit_rate@10": abs(float(official["accuracy@k"][10]) - entry["hit_rate@10"]),
        }
        if max(differences.values()) > 1e-6:
            raise AssertionError(f"Metric mismatch for {name}: {differences}")
        comparisons.append({"name": name, "differences": differences})
    output = {"sentence_transformers_revision": REVISION, "source_url": url,
              "source_sha256": hashlib.sha256(response.content).hexdigest(),
              "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
              "complete_benchmark": recorded["complete"], "comparisons": comparisons}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
