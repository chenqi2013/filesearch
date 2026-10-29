"""Official Qwen CPU FP32 reference on the already frozen RWKV/Qwen comparison."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("KMP_BLOCKTIME", "0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".codex-tmp/rwkv-qwen-comparison-20261029"))
    parser.add_argument("--model", type=Path, default=Path(".codex-tmp/qwen3-official-20261029"))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("comparison", root / "scripts/compare-rwkv-qwen.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    import torch
    from transformers import AutoModel, AutoTokenizer
    import transformers

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model_path = args.model.resolve()
    output = args.output.resolve()
    provenance = comparison.read(model_path / "provenance.json")
    assert comparison.digest(model_path / "model.safetensors") == provenance["model_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                     dtype=torch.float32, attn_implementation="sdpa").eval()
    load_seconds = time.perf_counter() - started
    dataset = comparison.read(output / "frozen-dataset.json")
    vectors, timings = {}, {}

    def infer(sequence):
        with torch.inference_mode():
            ids = torch.tensor([sequence], dtype=torch.long)
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0, -1]
            return torch.nn.functional.normalize(result.float(), dim=0).numpy()

    official_texts = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:Explain gravity",
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    sample = np.array([infer(tokenizer.encode(text)) for text in official_texts])
    scores = sample[:2] @ sample[2:].T
    expected = np.array([[.7645568251609802, .14142508804798126], [.13549736142158508, .5999549627304077]])
    sanity = dict(actual=scores.tolist(), official_example=expected.tolist(), max_abs_difference=float(np.abs(scores - expected).max()))
    comparison.save(output / "qwen-fp32-official-example-check.json", sanity)
    print("Official Qwen example parity: " + json.dumps(sanity), flush=True)
    queries = [case["query"] for case in dataset["cases"]]
    inputs = {key: [p["name"][:160] + "\n" + p["text"] for p in passages] for key, passages in dataset["passages"].items()}
    inputs.update(raw=queries, instructed=[comparison.INSTRUCTION.format(query) for query in queries])
    for label, texts in inputs.items():
        path = output / f"qwen-fp32-{label}.npz"
        cache_identity = dict(inputs_sha256=comparison.identity(texts), model_sha256=provenance["model_sha256"],
                              tokenizer_sha256=comparison.digest(model_path / "tokenizer.json"),
                              threads=args.threads, precision="CPU FP32", attention="PyTorch SDPA", batch_size=1,
                              torch=torch.__version__, transformers=transformers.__version__)
        started = time.perf_counter()
        sequences = [tokenizer.encode(text) for text in texts]
        tokenization_seconds = time.perf_counter() - started
        assert all(0 < len(seq) <= 1024 for seq in sequences), "Shared test exceeds common context; do not truncate silently"
        values = np.zeros((len(texts), 1024), dtype=np.float32)
        completed = np.zeros(len(texts), dtype=bool)
        durations = np.zeros(len(texts), dtype=np.float64)
        if path.exists():
            previous = np.load(path)
            assert json.loads(str(previous["identity"])) == cache_identity
            values, completed, durations = previous["vectors"], previous["completed"], previous["durations"]
        infer(sequences[0])
        last_report = time.perf_counter()
        for index, sequence in enumerate(sequences):
            if completed[index]:
                continue
            started = time.perf_counter()
            values[index] = infer(sequence)
            durations[index] = time.perf_counter() - started
            completed[index] = True
            if time.perf_counter() - last_report > 20 or completed.all():
                np.savez(path, identity=json.dumps(cache_identity), vectors=values, completed=completed, durations=durations)
                print(f"Qwen FP32 {label}: {completed.sum()}/{len(texts)}; inference {durations.sum():.1f}s", flush=True)
                last_report = time.perf_counter()
        assert np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
        vectors[f"qwen_fp32_{label}"] = values
        timings[label] = dict(**cache_identity, inference_seconds=float(durations.sum()),
                              latency_mean_ms=float(np.mean(durations) * 1000), latency_median_ms=float(np.median(durations) * 1000),
                              latency_p95_ms=float(np.percentile(durations, 95) * 1000), texts=len(texts),
                              tokenizer_seconds=tokenization_seconds, max_tokens=max(map(len, sequences)),
                              tokens=sum(map(len, sequences)), load_seconds=load_seconds,
                              checkpoint_bytes=(model_path / "model.safetensors").stat().st_size,
                              timing_note="PyTorch eager host calls with CPU SDPA, no production integration or serving optimization")
        comparison.save(output / f"qwen-fp32-{label}-timing.json", timings[label])
    result = dict(provenance=provenance, official_example_parity=sanity,
                  dataset_sha256=comparison.digest(output / "frozen-dataset.json"), timings=timings,
                  results=comparison.evaluate(dataset, vectors))
    comparison.save(output / "qwen-fp32-result.json", result)
    for name, values in result["results"].items():
        print(json.dumps(dict(model=name, summary=values["summary"])), flush=True)


if __name__ == "__main__":
    main()
