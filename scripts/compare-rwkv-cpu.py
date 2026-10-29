"""Controlled, uncached RWKV CPU throughput and retrieval comparison."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--baseline-encoder", type=Path)
    parser.add_argument("--candidate-encoder", type=Path)
    parser.add_argument("--candidate-batch-size", type=int)
    parser.add_argument("--candidate-token-budget", type=int, default=2048)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--candidate-affinity")
    parser.add_argument("--retrieval-batched", action="store_true",
                        help="Evaluate the actual baseline/candidate batch policies, rather than singletons")
    args = parser.parse_args()
    if (args.repeats < 1 or args.samples < 0 or args.batch_size < 1
            or (args.candidate_batch_size is not None and args.candidate_batch_size < 1)
            or args.candidate_token_budget < 1):
        parser.error("repeats and batch sizes/budgets must be positive; samples nonnegative")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).with_name("benchmark-long-txt.py"))
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    text = args.file.read_text(encoding="utf-8-sig")
    text = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
    passages = list(benchmark.windows(text, "direct"))
    selected = benchmark.select(passages, args.samples)
    texts = [args.file.name[:160] + "\n" + passage for passage in selected]
    args.root, args.inter_threads, args.spinning, args.profile = root, 1, False, None
    measured_batch_size = args.batch_size
    args.runtime = root / "src-tauri/binaries/onnxruntime.dll"
    runs, vectors = [], {}
    for repeat in range(0 if args.retrieval_only else args.repeats):
        order = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
        for name in order:
            args.rwkv_model = getattr(args, name).resolve()
            args.encoder = getattr(args, name + "_encoder")
            args.batch_size = (args.candidate_batch_size or measured_batch_size) if name == "candidate" else measured_batch_size
            args.batch_token_budget = args.candidate_token_budget if name == "candidate" else 2048
            args.cpu_affinity = args.candidate_affinity if name == "candidate" else None
            timing, embedding = benchmark.rwkv_run(args, texts)
            if (not np.isfinite(embedding).all()
                    or not np.allclose(np.linalg.norm(embedding, axis=1), 1, atol=1e-4)):
                raise ValueError("Invalid embeddings")
            vectors[name] = embedding
            runs.append(dict(name=name, repeat=repeat, **timing))
            print(json.dumps(runs[-1]), flush=True)

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    documents, queries = dataset["documents"], dataset["queries"]
    if not documents or not queries:
        raise ValueError("Empty retrieval dataset")
    ids = [doc["id"] for doc in documents]
    retrieval = {}
    for name in ["baseline", "candidate"]:
        args.rwkv_model = getattr(args, name).resolve()
        args.encoder = getattr(args, name + "_encoder")
        args.cpu_affinity = args.candidate_affinity if name == "candidate" else None
        args.batch_size = ((args.candidate_batch_size or measured_batch_size) if name == "candidate"
                           else measured_batch_size) if args.retrieval_batched else 1
        args.batch_token_budget = args.candidate_token_budget if name == "candidate" else 2048
        retrieval_timing, all_vectors = benchmark.rwkv_run(args, [row["text"] for row in documents + queries])
        if (not np.isfinite(all_vectors).all()
                or not np.allclose(np.linalg.norm(all_vectors, axis=1), 1, atol=1e-4)):
            raise ValueError("Invalid retrieval vectors")
        if args.retrieval_only:
            vectors[name] = all_vectors
        document_vectors, query_vectors = all_vectors[:len(documents)], all_vectors[len(documents):]
        ranking = np.argsort(-(query_vectors @ document_vectors.T), axis=1, kind="stable")
        relevant = [query["relevant"] if isinstance(query["relevant"], list) else [query["relevant"]]
                    for query in queries]
        ranks = [min(list(order).index(ids.index(doc)) + 1 for doc in positives)
                 for positives, order in zip(relevant, ranking)]
        recalls = [len(set(ids[index] for index in order[:10]) & set(positives)) / len(positives)
                   for positives, order in zip(relevant, ranking)]
        recalls3 = [len(set(ids[index] for index in order[:3]) & set(positives)) / len(positives)
                    for positives, order in zip(relevant, ranking)]
        retrieval[name] = dict(documents=len(documents), queries=len(queries),
                               top1=sum(rank == 1 for rank in ranks),
                               recall_at_3=float(np.mean(recalls3)),
                               hit_at_3=sum(rank <= 3 for rank in ranks) / len(ranks),
                               mrr_at_3=sum(1 / rank if rank <= 3 else 0 for rank in ranks) / len(ranks),
                               recall_at_10=float(np.mean(recalls)),
                               model_sha256=retrieval_timing["model_sha256"],
                               encoder_sha256=retrieval_timing["encoder_sha256"],
                               ranks=ranks)
    expected, actual = vectors["baseline"].astype(np.float64), vectors["candidate"].astype(np.float64)
    cosines = np.sum(expected * actual, axis=1) / (np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1))
    times = {name: statistics.median(run["inference_seconds"] for run in runs if run["name"] == name)
             for name in vectors} if runs else {}
    result = dict(file_sha256=benchmark.file_hash(args.file),
                  dataset_sha256=benchmark.file_hash(args.dataset),
                  dataset_path=str(args.dataset.resolve()),
                  total_windows=len(passages), measured_windows=0 if args.retrieval_only else len(texts),
                  full_document_measured=not args.retrieval_only and len(texts) == len(passages),
                  vector_comparison_source="retrieval documents and queries" if args.retrieval_only else "novel windows",
                  threads=args.threads, batch_size=measured_batch_size,
                  candidate_batch_size=args.candidate_batch_size or measured_batch_size,
                  candidate_token_budget=args.candidate_token_budget,
                  candidate_affinity=args.candidate_affinity,
                  retrieval_batched=args.retrieval_batched,
                  runtime_sha256=benchmark.file_hash(args.runtime),
                  median_seconds=times,
                  time_reduction=1 - times["candidate"] / times["baseline"] if times else None,
                  vector_cosine_min=float(cosines.min()), vector_cosine_mean=float(cosines.mean()),
                  max_vector_difference=float(np.abs(vectors["candidate"] - vectors["baseline"]).max()),
                  retrieval=retrieval, runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(args.output.with_suffix(".npz"), **vectors)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
