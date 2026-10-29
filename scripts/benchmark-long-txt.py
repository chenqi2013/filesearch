from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import struct
import subprocess
import threading
import time

import numpy as np


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RwkvTokenizer:
    def __init__(self, path):
        data = path.read_bytes()
        if data[:8] != b"RWKVTOK1":
            raise ValueError("Invalid RWKV vocabulary")
        count = struct.unpack_from("<I", data, 8)[0]
        self.tokens = {}
        self.lengths = defaultdict(set)
        offset = 12
        for token_id in range(count):
            length = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            token = data[offset:offset + length]
            offset += length
            if token_id and token:
                self.tokens[token] = token_id
                self.lengths[token[0]].add(length)
        self.lengths = {key: sorted(values, reverse=True) for key, values in self.lengths.items()}

    def encode(self, text):
        data = text.encode("utf-8")
        tokens = []
        offset = 0
        while offset < len(data) and len(tokens) < 1023:
            for length in self.lengths[data[offset]]:
                token = data[offset:offset + length]
                if len(token) == length and token in self.tokens:
                    tokens.append(self.tokens[token])
                    offset += length
                    break
        return tokens + [65535]


def windows(text, layout):
    seen = set()
    if layout == "legacy":
        chunks = []
        for start in range(0, len(text), 700):
            chunks.append(text[start:start + 800])
            if start + 800 >= len(text):
                break
    else:
        chunks = [text]
    for chunk in chunks:
        for start in range(0, len(chunk), 400):
            value = chunk[start:start + 480]
            if value not in seen:
                seen.add(value)
                yield value
            if start + 480 >= len(chunk):
                break


def select(values, count):
    if not count or count >= len(values):
        return values
    return [values[index * (len(values) - 1) // max(1, count - 1)] for index in range(count)]


def rwkv_run(args, texts):
    tokenizer = RwkvTokenizer(args.root / "assets/models/embedding-rwkv-tiny/rwkv_vocab.bin")
    started = time.perf_counter()
    sequences = [tokenizer.encode(text) for text in texts]
    tokenization_seconds = time.perf_counter() - started
    encoder = getattr(args, "encoder", None) or args.root / "target/release/examples/rwkv-embed.exe"
    command = [str(encoder),
               "--model", str(args.rwkv_model),
               "--cpu-threads", str(args.threads)]
    if not args.spinning:
        command.append("--no-spinning")
    environment = dict(os.environ, ORT_DYLIB_PATH=str(args.runtime))
    if args.profile:
        environment["FILESEARCH_ORT_PROFILE"] = str(args.profile.resolve())
    environment["FILESEARCH_ORT_INTER_THREADS"] = str(args.inter_threads)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               text=True, encoding="utf-8", env=environment,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    affinity = getattr(args, "cpu_affinity", None)
    if affinity:
        import psutil
        try:
            psutil.Process(process.pid).cpu_affinity([int(index) for index in affinity.split(",")])
        except Exception:
            process.kill()
            process.wait()
            raise
    replies = queue.Queue()

    def read():
        for line in process.stdout:
            replies.put(line)
        replies.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()

    def infer(batch):
        process.stdin.write(json.dumps({"tokens": batch}) + "\n")
        process.stdin.flush()
        reply = replies.get(timeout=180)
        if reply is None:
            raise RuntimeError("RWKV worker exited")
        return np.asarray(json.loads(reply), dtype=np.float32)

    try:
        started = time.perf_counter()
        infer([sequences[0]])
        load_and_warmup_seconds = time.perf_counter() - started
        grouped = defaultdict(list)
        for index, sequence in enumerate(sequences):
            grouped[len(sequence)].append((index, sequence))
        batches = []
        for length, group in sorted(grouped.items()):
            size = min(args.batch_size, max(1, getattr(args, "batch_token_budget", 2048) // length))
            batches.extend(group[start:start + size] for start in range(0, len(group), size))
        vectors = np.zeros((len(sequences), 768), dtype=np.float32)
        started = time.perf_counter()
        last_report = started
        for batch_index, batch in enumerate(batches):
            output = infer([sequence for _, sequence in batch])
            for (index, _), vector in zip(batch, output):
                vectors[index] = vector
            if time.perf_counter() - last_report > 20:
                print(f"RWKV {batch_index + 1}/{len(batches)} batches", flush=True)
                last_report = time.perf_counter()
        elapsed = time.perf_counter() - started
        import psutil
        memory = psutil.Process(process.pid).memory_info()
        return dict(inference_seconds=elapsed, tokenizer_seconds=tokenization_seconds,
                    load_and_warmup_seconds=load_and_warmup_seconds,
                    runtime=f"ONNX Runtime at {args.runtime} with Rust RWKV custom operator",
                    runtime_sha256=file_hash(args.runtime),
                    encoder_sha256=file_hash(encoder),
                    model_path=str(args.rwkv_model),
                    model_sha256=file_hash(args.rwkv_model),
                    process_memory_bytes=getattr(memory, "peak_wset", memory.rss),
                    tokens=sum(map(len, sequences)), calls=len(batches),
                    mean_tokens=float(np.mean(list(map(len, sequences))))), vectors
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=5)


def onnx_run(args, texts):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    if args.model == "e5":
        directory = args.root / "target/release/models/multilingual-e5-small"
        path = directory / "onnx/model.onnx"
        prefix = "passage: "
        maximum = 512
    else:
        directory = args.root / "assets/models/qwen3-embedding-0.6b"
        path = directory / "model_int8.onnx"
        prefix = ""
        maximum = 1024
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=maximum)
    started = time.perf_counter()
    sequences = [tokenizer.encode(prefix + text).ids for text in texts]
    tokenization_seconds = time.perf_counter() - started
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "1" if args.spinning else "0")
    started = time.perf_counter()
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    input_names = {value.name for value in session.get_inputs()}
    output_name = session.get_outputs()[0].name
    load_seconds = time.perf_counter() - started

    def infer(batch):
        length = max(map(len, batch))
        ids = np.full((len(batch), length), tokenizer.token_to_id("<|endoftext|>") or 0, dtype=np.int64)
        mask = np.zeros_like(ids)
        for index, sequence in enumerate(batch):
            ids[index, :len(sequence)] = sequence
            mask[index, :len(sequence)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        if "position_ids" in input_names:
            feeds["position_ids"] = np.broadcast_to(np.arange(length, dtype=np.int64), ids.shape)
        for name in input_names:
            if name.startswith("past_key_values."):
                feeds[name] = np.zeros((len(batch), 8, 0, 128), dtype=np.float32)
        output = session.run([output_name], feeds)[0]
        if args.model == "e5":
            expanded = mask.astype(np.float32)[..., None]
            vectors = (output * expanded).sum(axis=1) / expanded.sum(axis=1)
        else:
            vectors = output[np.arange(len(batch)), mask.sum(axis=1) - 1]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return vectors

    started = time.perf_counter()
    infer([sequences[0]])
    warmup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = []
    for start in range(0, len(sequences), args.batch_size):
        vectors.extend(infer(sequences[start:start + args.batch_size]))
        if start and start % 32 == 0:
            print(f"{args.model} {start}/{len(sequences)} windows", flush=True)
    elapsed = time.perf_counter() - started
    return dict(inference_seconds=elapsed, tokenizer_seconds=tokenization_seconds,
                runtime=f"Python ONNX Runtime {ort.__version__}",
                model_path=str(path), model_sha256=file_hash(path),
                load_and_warmup_seconds=load_seconds + warmup_seconds,
                tokens=sum(map(len, sequences)), truncated_windows=sum(len(sequence) == maximum for sequence in sequences),
                mean_tokens=float(np.mean(list(map(len, sequences))))), np.asarray(vectors)


def main():
    parser = argparse.ArgumentParser(description="CPU long-TXT benchmark; samples are explicitly extrapolated")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=["rwkv", "qwen", "e5"], default="rwkv")
    parser.add_argument("--layout", choices=["legacy", "direct"], default="legacy")
    parser.add_argument("--samples", type=int, default=96, help="0 runs the full document")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-token-budget", type=int, default=2048)
    parser.add_argument("--spinning", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--rwkv-model", type=Path, help="Explicit RWKV candidate; production model is the default")
    parser.add_argument("--runtime", type=Path, help="Explicit Rust ONNX Runtime DLL; bundled runtime is the default")
    parser.add_argument("--encoder", type=Path, help="Explicit Rust encoder executable")
    parser.add_argument("--cpu-affinity", help="Experiment only: logical CPUs for this encoder process, comma separated")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.rwkv_model = (args.rwkv_model or args.root / "assets/models/embedding-rwkv-tiny/model.onnx").resolve()
    args.runtime = (args.runtime or args.root / "src-tauri/binaries/onnxruntime.dll").resolve()
    if args.samples < 0 or args.batch_size < 1 or args.threads < 1 or args.batch_token_budget < 1:
        parser.error("samples must be nonnegative; batch size and threads must be positive")
    data = args.file.read_bytes()
    try:
        text = data.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = data.decode("gb18030")
        encoding = "gb18030"
    text = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
    passages = list(windows(text, args.layout))
    selected = select(passages, args.samples)
    texts = [args.file.name[:160] + "\n" + passage for passage in selected]
    result = dict(model=args.model, layout=args.layout, file_sha256=hashlib.sha256(data).hexdigest(),
                  bytes=len(data), encoding=encoding, characters=len(text), total_windows=len(passages),
                  measured_windows=len(texts), full_document_measured=len(texts) == len(passages),
                  threads=args.threads, inter_threads=args.inter_threads,
                  batch_size=args.batch_size, spinning=args.spinning,
                  batch_token_budget=args.batch_token_budget,
                  cpu_affinity=args.cpu_affinity,
                  platform=platform.platform(), python_version=platform.python_version(),
                  device="CPU", sqlite_and_text_index_included=False)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    measured, vectors = rwkv_run(args, texts) if args.model == "rwkv" else onnx_run(args, texts)
    result.update(measured)
    result["estimated_full_inference_seconds"] = measured["inference_seconds"] * len(passages) / len(texts)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid embedding vectors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(args.output.with_suffix(".npy"), vectors)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
