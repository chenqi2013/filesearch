from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Fresh RWKV-only local retrieval evaluation")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "rwkv_benchmark", Path(__file__).with_name("benchmark-rwkv-nanobeir.py")
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    sys.path.insert(0, str(args.repository / "package" / "src"))
    from rwkv_emb.tokenizer import RWKVTokenizer

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    documents, queries = dataset["documents"], dataset["queries"]
    document_ids = [document["id"] for document in documents]
    if not queries or len(documents) < 3 or len(set(document_ids)) != len(documents):
        raise ValueError("Expected queries and at least three uniquely identified documents")
    if any(query["relevant"] not in document_ids for query in queries):
        raise ValueError("Relevant document missing from corpus")
    tokenizer = RWKVTokenizer()
    encoder = benchmark.RustEncoder(args.executable, args.model, args.runtime)
    try:
        document_vectors, document_timing = benchmark.encode(
            [document["text"] for document in documents], tokenizer, encoder, 4, 1024, "Documents"
        )
        query_vectors, query_timing = benchmark.encode(
            [query["text"] for query in queries], tokenizer, encoder, 4, 1024, "Queries"
        )
        scores = query_vectors @ document_vectors.T
        rankings = torch.argsort(scores, dim=1, descending=True, stable=True).tolist()
        details = []
        for query_index, (query, ranking) in enumerate(zip(queries, rankings)):
            rank = next(position + 1 for position, index in enumerate(ranking)
                        if document_ids[index] == query["relevant"])
            details.append({
                "query": query["text"], "relevant": query["relevant"], "rank": rank,
                "top3": [{"document_id": document_ids[index], "cosine": float(scores[query_index, index])}
                         for index in ranking[:3]],
            })
        top1_hits = sum(detail["rank"] == 1 for detail in details)
        top3_hits = sum(detail["rank"] <= 3 for detail in details)
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": "EmbeddingRWKV Tiny (corrected Rust WKV + ONNX)",
            "dataset_sha256": benchmark.sha256(args.dataset),
            "onnx_sha256": benchmark.sha256(args.model),
            "executable_sha256": benchmark.sha256(args.executable),
            "runtime_sha256": benchmark.sha256(args.runtime),
            "protocol": {"device": "CPU", "batch_size": 4, "max_tokens": 1024,
                         "pooling": "single EOS", "instruction": None, "reranker": False,
                         "similarity": "normalized cosine", "cached_vectors": False,
                         "relevant_documents_per_query": 1},
            "documents": len(documents), "queries": len(queries),
            "top1_hits": top1_hits, "top1": top1_hits / len(queries),
            "top3_hits": top3_hits, "recall@3": top3_hits / len(queries),
            "mrr@3": sum(1 / detail["rank"] if detail["rank"] <= 3 else 0
                         for detail in details) / len(queries),
            "document_timing": document_timing, "query_timing": query_timing,
            "details": details,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
