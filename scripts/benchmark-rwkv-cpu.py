from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import queue
import sqlite3
import statistics
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description="Compare CPU scheduling on identical indexed RWKV windows")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 2, 1])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "rwkv_utils", args.repository / "package/src/rwkv_emb/reference/utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.TRIE_TOKENIZER(str(args.repository / "package/src/rwkv_emb/reference/rwkv_vocab_v20230424.txt"))
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT d.name, p.text FROM semantic_passages p JOIN documents d ON d.id=p.document_id ORDER BY p.id"
        ).fetchall()
    if not rows:
        raise ValueError("No indexed semantic windows")
    count = min(max(args.samples, 1), len(rows))
    selected = [rows[index * (len(rows) - 1) // max(count - 1, 1)] for index in range(count)]
    tokens = [tokenizer.encode(name[:160] + "\n" + text)[:1023] + [65535] for name, text in selected]
    baseline = None
    results = []
    configurations = [(4, True)] + [(threads, False) for threads in args.threads] + [(4, True)]
    for threads, spinning in configurations:
        command = [str(args.executable.resolve()), "--model", str(args.model.resolve()), "--cpu-threads", str(threads)]
        if not spinning:
            command.append("--no-spinning")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, encoding="utf-8",
                                   env=dict(os.environ, ORT_DYLIB_PATH=str(args.runtime.resolve())),
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = queue.Queue()

        def read_output():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        def embed(sequence):
            process.stdin.write(json.dumps({"tokens": [sequence]}) + "\n")
            process.stdin.flush()
            line = lines.get(timeout=180)
            if line is None:
                raise RuntimeError("Encoder exited")
            return json.loads(line)[0]

        try:
            embed(tokens[0])
            durations, vectors = [], []
            for sequence in tokens:
                started = time.perf_counter()
                vectors.append(embed(sequence))
                durations.append(time.perf_counter() - started)
            if baseline is None:
                baseline = vectors
            difference = max(abs(left - right) for expected, actual in zip(baseline, vectors)
                             for left, right in zip(expected, actual))
            result = {"threads": threads, "spinning": spinning, "seconds": sum(durations),
                      "median_ms": statistics.median(durations) * 1000,
                      "max_vector_difference": difference}
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join(timeout=5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"samples": count, "tokens": sum(map(len, tokens)),
                                      "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
