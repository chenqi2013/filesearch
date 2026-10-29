"""Official RWKV C FP32 reference for the frozen RWKV/Qwen dataset."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".codex-tmp/rwkv-qwen-comparison-20261029"))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    comparison = load("comparison", root / "scripts/compare-rwkv-qwen.py")
    official = load("official_rwkv", root / "scripts/compare-official-rwkv.py")
    dataset = comparison.read(output / "frozen-dataset.json")
    vectors, timings = {}, {}
    queries = [case["query"] for case in dataset["cases"]]
    inputs = {key: [p["name"][:160] + "\n" + p["text"] for p in passages] for key, passages in dataset["passages"].items()}
    inputs.update(raw=queries, instructed=[comparison.INSTRUCTION.format(query) for query in queries])
    runtime = official.OfficialCPU(root / ".codex-tmp/official-rwkv-20261028", args.threads)
    try:
        for label, texts in inputs.items():
            path = output / f"rwkv-fp32-{label}.npz"
            cache_identity = dict(inputs_sha256=comparison.identity(texts), model_sha256=comparison.digest(official.WEIGHTS),
                                  source_revision=official.REVISION, precision="Official C CPU FP32", threads=args.threads,
                                  dll_sha256=comparison.digest(root / ".codex-tmp/official-rwkv-20261028/rwkv_emb.dll"), batch_size=1)
            sequences = [runtime.tokenize(text) for text in texts]
            assert all(len(seq) <= 1023 for seq in sequences)
            sequences = [seq + [65535] for seq in sequences]
            values = np.zeros((len(texts), 768), dtype=np.float32)
            completed = np.zeros(len(texts), dtype=bool)
            durations = np.zeros(len(texts), dtype=np.float64)
            if path.exists():
                previous = np.load(path)
                assert json.loads(str(previous["identity"])) == cache_identity
                values, completed, durations = previous["vectors"], previous["completed"], previous["durations"]
            runtime.tokens([sequences[0]])
            last_report = time.perf_counter()
            for index, sequence in enumerate(sequences):
                if completed[index]:
                    continue
                started = time.perf_counter()
                values[index] = runtime.tokens([sequence])[0]
                durations[index] = time.perf_counter() - started
                completed[index] = True
                if time.perf_counter() - last_report > 20 or completed.all():
                    np.savez(path, identity=json.dumps(cache_identity), vectors=values, completed=completed, durations=durations)
                    print(f"RWKV official FP32 {label}: {completed.sum()}/{len(texts)}; inference {durations.sum():.1f}s", flush=True)
                    last_report = time.perf_counter()
            assert np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
            vectors[f"rwkv_fp32_{label}"] = values
            timings[label] = dict(**cache_identity, inference_seconds=float(durations.sum()),
                                  texts=len(texts), tokens=sum(map(len, sequences)), max_tokens=max(map(len, sequences)))
            comparison.save(output / f"rwkv-fp32-{label}-timing.json", timings[label])
    finally:
        runtime.close()
    results = comparison.evaluate(dataset, vectors)
    comparison.save(output / "rwkv-fp32-result.json", dict(dataset_sha256=comparison.digest(output / "frozen-dataset.json"), timings=timings, results=results))
    for name, result in results.items():
        print(json.dumps(dict(model=name, summary=result["summary"])), flush=True)


if __name__ == "__main__":
    main()
