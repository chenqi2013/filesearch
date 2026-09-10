from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import pyarrow.parquet as parquet
import requests
import torch
from torch.nn import functional as F


DATASET_REVISION = "beb106fbcfaa599c508c667041bf8c85fd78736b"
DATASET_ID = "sentence-transformers/NanoBEIR-en"
DATASETS = ["SciFact", "ArguAna", "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018",
            "HotpotQA", "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "Touche2020"]


class RustEncoder:
    def __init__(self, executable, model, runtime):
        environment = dict(os.environ, ORT_DYLIB_PATH=str(runtime.resolve()))
        self.process = subprocess.Popen(
            [str(executable.resolve()), "--model", str(model.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.peak_rss = 0
        self.finished = threading.Event()
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _monitor(self):
        process = psutil.Process(self.process.pid)
        while not self.finished.wait(0.1):
            try:
                self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            except psutil.NoSuchProcess:
                break

    def embed(self, tokens):
        self.process.stdin.write(json.dumps({"tokens": tokens}) + "\n")
        self.process.stdin.flush()
        line = self.lines.get(timeout=180)
        if line is None:
            raise RuntimeError(f"Rust encoder exited: {self.process.poll()}")
        vectors = torch.tensor(json.loads(line), dtype=torch.float32)
        if vectors.shape != (len(tokens), 768) or not torch.isfinite(vectors).all():
            raise ValueError("Invalid Rust embedding output")
        return vectors

    def close(self):
        self.finished.set()
        self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=5)
        self.monitor.join(timeout=5)


def load_audit():
    spec = importlib.util.spec_from_file_location("rwkv_audit", Path(__file__).with_name("audit-embedding-rwkv.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_dataset(root, name):
    rows, hashes = {}, {}
    for kind in ["corpus", "queries", "qrels"]:
        filename = f"Nano{name}-00000-of-00001.parquet"
        path = root / DATASET_REVISION / kind / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{kind}/{filename}"
            last_error = None
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=90)
                    response.raise_for_status()
                    temporary = path.with_suffix(".download")
                    temporary.write_bytes(response.content)
                    parquet.read_metadata(temporary)
                    temporary.replace(path)
                    last_error = None
                    break
                except (requests.RequestException, OSError) as error:
                    last_error = error
                    time.sleep(attempt + 1)
            if last_error:
                raise last_error
        rows[kind] = parquet.read_table(path).to_pylist()
        hashes[kind] = sha256(path)
    return rows, hashes


def encode(texts, tokenizer, encoder, batch_size, max_tokens, label):
    started = time.perf_counter()
    tokenized = [tokenizer.encode(text, add_eos=False)[:max_tokens - 1] + [65535] for text in texts]
    tokenization_seconds = time.perf_counter() - started
    order = sorted(range(len(texts)), key=lambda index: len(tokenized[index]))
    vectors = torch.empty((len(texts), 768))
    last_report = time.perf_counter()
    inference_started = time.perf_counter()
    for offset in range(0, len(order), batch_size):
        indexes = order[offset:offset + batch_size]
        vectors[indexes] = encoder.embed([tokenized[index] for index in indexes])
        if time.perf_counter() - last_report > 25 or offset + batch_size >= len(order):
            print(f"{label}: {min(offset + batch_size, len(order))}/{len(order)} "
                  f"{time.perf_counter() - inference_started:.1f}s", flush=True)
            last_report = time.perf_counter()
    return vectors, {"tokenization_seconds": tokenization_seconds,
                     "inference_seconds": time.perf_counter() - inference_started,
                     "tokens": sum(map(len, tokenized))}


def verify(encoder, args, audit, tokenizer):
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    exporter = audit.load_exporter()
    candidate = exporter.EmbeddingRwkvTiny(state).eval()
    namespace, _ = audit.load_reference(args.repository)
    result = audit.verify(namespace, state, candidate)
    texts = ["Local document search", "EmbeddingRWKV Tiny test", "", "network security " * 80]
    ids = [tokenizer.encode(text) for text in texts]
    maximum_length = max(map(len, ids))
    padded = torch.tensor([[0] * (maximum_length - len(tokens)) + tokens for tokens in ids])
    with torch.inference_mode():
        expected = F.normalize(candidate(padded), dim=1)
    actual = encoder.embed(ids)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-3)
    result["rust_onnx_max_error"] = float((actual - expected).abs().max())
    result["rust_onnx_min_cosine"] = float(F.cosine_similarity(actual, expected).min())
    print(json.dumps({"verification": result}), flush=True)
    return result


def evaluate(rows, name, tokenizer, encoder, audit, args):
    documents, queries = rows["corpus"], rows["queries"]
    query_ids = {row["_id"] for row in queries}
    document_ids = {row["_id"] for row in documents}
    if len(document_ids) != len(documents) or len(query_ids) != len(queries):
        raise ValueError("Duplicate dataset identifiers")
    qrels = {}
    for row in rows["qrels"]:
        if row.get("score", 1) > 0:
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    if not all(qrels.get(query_id) for query_id in query_ids):
        raise ValueError("Queries without positive judgments")
    if not set().union(*qrels.values()).issubset(document_ids):
        raise ValueError("Judged document missing from corpus")
    document_texts = [(row.get("title", "") + " " + row["text"]).strip() for row in documents]
    query_texts = [row["text"] for row in queries]
    document_vectors, document_timing = encode(document_texts, tokenizer, encoder, args.batch_size, args.max_tokens, name + " corpus")
    query_vectors, query_timing = encode(query_texts, tokenizer, encoder, args.batch_size, args.max_tokens, name + " queries")
    started = time.perf_counter()
    scores = query_vectors @ document_vectors.T
    result = audit.metrics(scores, documents, queries, qrels)
    result.update(name=name, documents=len(documents), queries=len(queries),
                  document_timing=document_timing, query_timing=query_timing,
                  retrieval_seconds=time.perf_counter() - started,
                  rust_process_peak_rss_mb=encoder.peak_rss / 1024**2)
    cache = args.output.parent / (args.output.stem + "-vectors")
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{name}.npz", documents=document_vectors.numpy(), queries=query_vectors.numpy())
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--local-eval", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_tokens < 2:
        parser.error("batch-size must be positive; max-tokens must be at least 2")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("datasets must not contain duplicates")
    sys.path.insert(0, str(args.repository / "package" / "src"))
    from rwkv_emb.tokenizer import RWKVTokenizer

    tokenizer = RWKVTokenizer()
    audit = load_audit()
    result = {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
              "upstream_revision": audit.UPSTREAM_REVISION, "checkpoint_sha256": sha256(args.checkpoint),
              "onnx_sha256": sha256(args.model), "executable_sha256": sha256(args.executable),
              "protocol": {"profile": "rwkv7-tiny-corrected-single-eos-v2", "device": "CPU",
                           "max_tokens": args.max_tokens, "batch_size": args.batch_size,
                           "instruction": None, "pooling": "single EOS", "reranker": False,
                           "official_mteb_runner": False, "qrels": "binary", "similarity": "cosine"},
              "results": [], "complete": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for field in ["dataset_revision", "upstream_revision", "checkpoint_sha256", "onnx_sha256",
                      "executable_sha256", "protocol"]:
            if previous[field] != result[field]:
                raise ValueError(f"Cannot resume with changed {field}")
        result["results"] = [entry for entry in previous["results"] if entry["name"] in args.datasets]
        for entry in result["results"]:
            vector_path = args.output.parent / (args.output.stem + "-vectors") / (entry["name"] + ".npz")
            if not vector_path.is_file():
                raise ValueError(f"Missing cached vectors for {entry['name']}")

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(args.output)

    encoder = RustEncoder(args.executable, args.model, args.runtime)
    try:
        result["verification"] = verify(encoder, args, audit, tokenizer)
        save()
        for name in args.datasets:
            if any(entry["name"] == name for entry in result["results"]):
                print(f"Reusing completed dataset {name}", flush=True)
                continue
            rows, hashes = download_dataset(args.data, name)
            entry = evaluate(rows, name, tokenizer, encoder, audit, args)
            entry["parquet_sha256"] = hashes
            result["results"].append(entry)
            save()
        if args.local_eval:
            local = json.loads(args.local_eval.read_text(encoding="utf-8"))
            rows = {"corpus": [{"_id": row["id"], "text": row["text"]} for row in local["documents"]],
                    "queries": [{"_id": str(index), "text": row["text"]} for index, row in enumerate(local["queries"])],
                    "qrels": [{"query-id": str(index), "corpus-id": row["relevant"]} for index, row in enumerate(local["queries"])]}
            result["local_diagnostic"] = evaluate(rows, "ChineseDiagnostic", tokenizer, encoder, audit, args)
            result["local_diagnostic"]["sha256"] = sha256(args.local_eval)
        result["macro_ndcg@10"] = sum(row["ndcg@10"] for row in result["results"]) / len(result["results"])
        result["complete"] = True
        result["full_nanobeir"] = set(args.datasets) == set(DATASETS)
        save()
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
