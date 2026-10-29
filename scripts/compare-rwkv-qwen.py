"""CPU embedding retrieval comparison on frozen, shared passages; no production changes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import time
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Instruct: Given a query, retrieve documents that answer the query\nQuery: {}"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def normalized(value):
    return "".join(value.lower().split())


def freeze(output, database):
    benchmark = module("comparison_benchmark", ROOT / "scripts/benchmark-long-txt.py")
    verifier = module("comparison_storage", ROOT / "scripts/verify-search-ranking.py")
    before = verifier.vector_state(database)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        documents = [dict(id=row[0], name=row[1]) for row in db.execute("SELECT id,name FROM documents ORDER BY id")]
        excluded = [doc for doc in documents if doc["name"] == "小说——超级龙宠.txt"]
        documents = [doc for doc in documents if doc not in excluded]
        names = {doc["id"]: doc["name"] for doc in documents}
        real = [dict(id=str(row[0]), document=row[1], name=names[row[1]], text=row[2])
                for row in db.execute("SELECT id,document_id,text FROM semantic_passages ORDER BY id") if row[1] in names]
    cases = []
    for filename, split in [("semantic-generalization-20261027.json", "calibration"),
                            ("semantic-holdout-20261028.json", "holdout")]:
        suite = read(ROOT / "scripts/fixtures" / filename)
        for kind in ["new_positive", "controls"]:
            for group, query, evidence in suite[kind]:
                if group == "novel":
                    continue
                targets = suite["target_groups"][group]
                assert any(normalized(evidence) in normalized(p["text"]) for p in real if p["name"] in targets), (query, evidence)
                cases.append(dict(corpus="real", split=split, kind="positive", group=group,
                                  query=query, targets=targets, evidence=evidence))
        for query in suite["negative"]:
            cases.append(dict(corpus="real", split=split, kind="negative", group="negative",
                              query=query, targets=[], evidence=""))
    fixture = read(ROOT / "scripts/fixtures/rwkv-qwen-cases-20261029.json")
    synthetic = []
    for doc, text in fixture["documents"]:
        for position, passage in enumerate(benchmark.windows(text, "direct")):
            synthetic.append(dict(id=f"{doc}:{position}", document=doc, name=f"sample-{doc}.txt", text=passage))
    for group, query, targets, evidence in fixture["queries"]:
        target_names = [f"sample-{doc}.txt" for doc in targets]
        assert any(normalized(evidence) in normalized(p["text"]) for p in synthetic if p["name"] in target_names), (query, evidence)
        cases.append(dict(corpus="synthetic", split="new_cases", kind="positive", group=group,
                          query=query, targets=target_names, evidence=evidence))
    cases.extend(dict(corpus="synthetic", split="new_cases", kind="negative", group="negative",
                      query=query, targets=[], evidence="") for query in fixture["negative"])
    passages = dict(real=real, synthetic=synthetic)
    dataset = dict(passages=passages, cases=cases, source_state=before,
                   source_database=str(database), excluded_documents=excluded,
                   scope="Full windows of 13 non-novel source documents; novel excluded from both models, not sampled. Separate synthetic corpus.",
                   fixture_sha256=digest(ROOT / "scripts/fixtures/rwkv-qwen-cases-20261029.json"))
    path = output / "frozen-dataset.json"
    if path.exists():
        assert read(path) == dataset, "Dataset changed after freeze"
    else:
        save(path, dataset)
    return dataset


class Qwen:
    def __init__(self, threads):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.path = ROOT / "assets/models/qwen3-embedding-0.6b/model_int8.onnx"
        self.tokenizer = Tokenizer.from_file(str(self.path.with_name("tokenizer.json")))
        # Batch one: no padding; pool the final valid token as in Qwen's official example.
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        started = time.perf_counter()
        self.session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        self.load_seconds = time.perf_counter() - started
        self.inputs = self.session.get_inputs()
        self.output = self.session.get_outputs()[0].name
        self.runtime = ort.__version__

    def encode_one(self, sequence):
        ids = np.asarray([sequence], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids),
                 "position_ids": np.arange(len(sequence), dtype=np.int64)[None, :]}
        for value in self.inputs:
            if value.name.startswith("past_key_values."):
                feeds[value.name] = np.empty((1, 8, 0, 128), dtype=np.float32)
        tokens = self.session.run([self.output], feeds)[0]
        vector = tokens[0, -1].astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def run(self, texts, output, label):
        cache = output / f"qwen-{label}.npz"
        cache_identity = dict(inputs_sha256=identity(texts), model_sha256=digest(self.path),
                              tokenizer_sha256=digest(self.path.with_name("tokenizer.json")),
                              runtime=self.runtime, maximum_tokens=1024, pooling="last valid token", batch_size=1)
        started = time.perf_counter()
        sequences = [self.tokenizer.encode(text).ids for text in texts]
        tokenization_seconds = time.perf_counter() - started
        assert all(0 < len(seq) <= 1024 for seq in sequences), "Shared test exceeds context; never silently truncate"
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        completed = np.zeros(len(texts), dtype=bool)
        durations = np.zeros(len(texts), dtype=np.float64)
        if cache.exists():
            previous = np.load(cache)
            assert json.loads(str(previous["identity"])) == cache_identity
            vectors, completed, durations = previous["vectors"], previous["completed"], previous["durations"]
        started = time.perf_counter()
        self.encode_one(sequences[0])
        warmup_seconds = time.perf_counter() - started
        last_report = time.perf_counter()
        for index, sequence in enumerate(sequences):
            if completed[index]:
                continue
            started = time.perf_counter()
            vectors[index] = self.encode_one(sequence)
            durations[index] = time.perf_counter() - started
            completed[index] = True
            if time.perf_counter() - last_report > 20 or completed.all():
                np.savez(cache, identity=json.dumps(cache_identity), vectors=vectors, completed=completed, durations=durations)
                print(f"Qwen {label}: {completed.sum()}/{len(texts)}; measured {durations.sum():.1f}s", flush=True)
                last_report = time.perf_counter()
        report = dict(**cache_identity, inference_seconds=float(durations.sum()),
                      latency_median_ms=float(np.median(durations) * 1000),
                      latency_p95_ms=float(np.percentile(durations, 95) * 1000),
                      tokenizer_seconds=tokenization_seconds, load_seconds=self.load_seconds,
                      warmup_seconds=warmup_seconds, texts=len(texts), tokens=sum(map(len, sequences)),
                      max_tokens=max(map(len, sequences)), model_bytes=self.path.stat().st_size,
                      timing_note="Uncached model calls, Python tokenization and loading reported separately")
        save(output / f"qwen-{label}-timing.json", report)
        return vectors, report


def rwkv(texts, args, label):
    benchmark = module("rwkv_comparison_benchmark", ROOT / "scripts/benchmark-long-txt.py")
    path = args.output / f"rwkv-{label}.npz"
    timing_path = args.output / f"rwkv-{label}-timing.json"
    model = ROOT / "assets/models/embedding-rwkv-tiny/model.onnx"
    cache_identity = dict(inputs_sha256=identity(texts), model_sha256=digest(model), threads=args.threads,
                          batch_size=1, encoder_sha256=digest(ROOT / "target/release/examples/rwkv-embed.exe"))
    if path.exists():
        previous = np.load(path)
        assert json.loads(str(previous["identity"])) == cache_identity
        return previous["vectors"], read(timing_path)
    parameters = SimpleNamespace(root=ROOT, rwkv_model=model, runtime=ROOT / "src-tauri/binaries/onnxruntime.dll",
                                 threads=args.threads, inter_threads=1, spinning=False, profile=None,
                                 batch_size=1, batch_token_budget=2048)
    timing, vectors = benchmark.rwkv_run(parameters, texts)
    report = dict(**timing, inputs_sha256=identity(texts), threads=args.threads, batch_size=1,
                  texts=len(texts), model_bytes=model.stat().st_size,
                  mean_ms=timing["inference_seconds"] / len(texts) * 1000,
                  timing_note="Rust inference worker request roundtrip; tokenizer/loading excluded")
    save(timing_path, report)
    np.savez(path, identity=json.dumps(cache_identity), vectors=vectors)
    return vectors, report


def rank(case, passages, scores):
    order = np.argsort(-scores, kind="stable")
    ordered_docs = []
    seen = set()
    for index in order:
        passage = passages[index]
        if passage["document"] not in seen:
            seen.add(passage["document"])
            ordered_docs.append(dict(name=passage["name"], score=float(scores[index]), passage_id=passage["id"],
                                     text=passage["text"]))
    target_rank = next((position for position, doc in enumerate(ordered_docs, 1) if doc["name"] in case["targets"]), None)
    evidence_indices = {i for i, p in enumerate(passages) if p["name"] in case["targets"]
                        and normalized(case["evidence"]) in normalized(p["text"])} if case["kind"] == "positive" else set()
    evidence_rank = next((position for position, index in enumerate(order, 1) if index in evidence_indices), None)
    return dict(**case, target_rank=target_rank, evidence_rank=evidence_rank,
                best_score=float(scores[order[0]]),
                target_score=max((float(scores[i]) for i, p in enumerate(passages) if p["name"] in case["targets"]), default=None),
                top_documents=ordered_docs[:10],
                top_passages=[dict(name=passages[i]["name"], text=passages[i]["text"], score=float(scores[i])) for i in order[:3]])


def calibration_threshold(rows):
    positive = [row for row in rows if row["kind"] == "positive"]
    negative = [row for row in rows if row["kind"] == "negative"]
    values = sorted({row["best_score"] for row in rows})
    thresholds = [values[0] - 1e-6, *[(a + b) / 2 for a, b in zip(values, values[1:])], values[-1] + 1e-6]
    def utility(threshold):
        sensitivity = sum(row["best_score"] >= threshold for row in positive) / len(positive)
        specificity = sum(row["best_score"] < threshold for row in negative) / len(negative)
        return (sensitivity + specificity) / 2, specificity, threshold
    score, specificity, threshold = max(map(utility, thresholds))
    return dict(threshold=threshold, balanced_accuracy=score, specificity=specificity,
                criterion="Maximize answerable-vs-unanswerable balanced accuracy on real calibration only; ties prefer specificity then higher cutoff")


def summarize(rows, threshold):
    positive = [row for row in rows if row["kind"] == "positive"]
    negative = [row for row in rows if row["kind"] == "negative"]
    ranks = [row["target_rank"] for row in positive]
    evidence = [row["evidence_rank"] for row in positive]
    area = (sum((p["best_score"] > n["best_score"]) + 0.5 * (p["best_score"] == n["best_score"])
                for p in positive for n in negative) / (len(positive) * len(negative))) if positive and negative else None
    return dict(positive=len(positive), top1=sum(rank == 1 for rank in ranks),
                hit3=sum(rank is not None and rank <= 3 for rank in ranks),
                mrr10=sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / len(positive) if positive else None,
                evidence_top1=sum(rank == 1 for rank in evidence), evidence_hit3=sum(rank is not None and rank <= 3 for rank in evidence),
                negatives=len(negative), answerability_auroc=area,
                calibrated_cutoff=threshold,
                negative_rejected=sum(row["best_score"] < threshold for row in negative),
                positive_target_kept=sum(row["target_score"] >= threshold for row in positive),
                correct_top1_kept=sum(row["target_rank"] == 1 and row["target_score"] >= threshold for row in positive))


def evaluate(dataset, vectors):
    cases = dataset["cases"]
    result = {}
    for name in sorted(key for key in vectors if key.endswith(("_raw", "_instructed"))):
        model, view = name.rsplit("_", 1)
        queries = vectors[f"{model}_{view}"]
        rows = []
        for corpus, passages in dataset["passages"].items():
            scores = queries @ vectors[f"{model}_{corpus}"].T
            rows.extend(rank(case, passages, scores[index]) for index, case in enumerate(cases) if case["corpus"] == corpus)
        calibration = calibration_threshold([row for row in rows if row["corpus"] == "real" and row["split"] == "calibration"])
        threshold = calibration["threshold"]
        result[name] = dict(calibration=calibration, summary={}, rows=rows)
        for corpus, split in [("real", "calibration"), ("real", "holdout"), ("synthetic", "new_cases")]:
            selected = [row for row in rows if row["corpus"] == corpus and row["split"] == split]
            result[name]["summary"][corpus + "_" + split] = summarize(selected, threshold)
        result[name]["groups"] = {
            corpus: {group: summarize([row for row in rows if row["corpus"] == corpus and row["group"] == group], threshold)
                     for group in sorted({row["group"] for row in rows if row["corpus"] == corpus and row["kind"] == "positive"})}
            for corpus in dataset["passages"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / ".codex-tmp/semantic-generalization-20261027/run1/data/search.db")
    parser.add_argument("--output", type=Path, default=ROOT / ".codex-tmp/rwkv-qwen-comparison-20261029")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = freeze(args.output, args.database.resolve())
    print(json.dumps(dict(scope=dataset["scope"], passages={key: len(value) for key, value in dataset["passages"].items()},
                          cases=dict(Counter((case["corpus"] + ":" + case["split"] + ":" + case["kind"]) for case in dataset["cases"]))), ensure_ascii=False), flush=True)
    if args.freeze_only:
        return
    vectors, timings = {}, {}
    queries = [case["query"] for case in dataset["cases"]]
    inputs = {key: [p["name"][:160] + "\n" + p["text"] for p in passages] for key, passages in dataset["passages"].items()}
    inputs.update(raw=queries, instructed=[INSTRUCTION.format(query) for query in queries])
    for label, texts in inputs.items():
        vectors[f"rwkv_{label}"], timings[f"rwkv_{label}"] = rwkv(texts, args, label)
        print("RWKV finished " + label, flush=True)
    qwen = Qwen(args.threads)
    for label, texts in inputs.items():
        vectors[f"qwen_{label}"], timings[f"qwen_{label}"] = qwen.run(texts, args.output, label)
    for name, values in vectors.items():
        assert np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4), name
    verifier = module("final_comparison_storage", ROOT / "scripts/verify-search-ranking.py")
    assert verifier.vector_state(args.database.resolve()) == dataset["source_state"], "Source index changed"
    results = evaluate(dataset, vectors)
    report = dict(dataset_sha256=digest(args.output / "frozen-dataset.json"), instruction=INSTRUCTION,
                  environment=dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                                   logical_cpus=os.cpu_count(), threads=args.threads, backend="CPU only; 1 interop thread; spinning disabled"),
                  source_unchanged=True, corpus_scope=dataset["scope"], timings=timings, results=results,
                  limitations=["Developer-authored cases, not customer blind accuracy", "Qwen local INT8 ONNX export, not official BF16 parity validation",
                               "RWKV partial INT8 0.1B and Qwen INT8 0.6B have different sizes and runtimes",
                               "Raw cosine ranking; no BM25, production gates, anchors or reranker", "Calibration fitted only on real calibration, never on holdout or new cases"])
    save(args.output / "result.json", report)
    for name, value in results.items():
        print(json.dumps(dict(model=name, summary=value["summary"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
