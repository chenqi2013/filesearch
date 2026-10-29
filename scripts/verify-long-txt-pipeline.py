from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import time
from urllib.request import Request, urlopen

import psutil


def main():
    parser = argparse.ArgumentParser(description="Isolated RWKV TXT indexing, pause and restart-cache verification")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--interrupt-after-windows", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, help="Isolated candidate model directory")
    parser.add_argument("--executable", type=Path, help="Explicit release search-core executable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_dir = (args.model_dir or root / "assets/models/embedding-rwkv-tiny").resolve()
    executable = (args.executable or root / "target/release/search-core.exe").resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    target = source / args.file.name
    if args.full:
        shutil.copy2(args.file, target)
    else:
        text = args.file.read_text(encoding="utf-8-sig")
        sample = "\n".join(text[index * (len(text) - 480) // 63:][:480] for index in range(64))
        target.write_text(sample, encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    environment = dict(os.environ, ORT_DYLIB_PATH=str(root / "src-tauri/binaries/onnxruntime.dll"), RUST_LOG="info")
    environment.pop("FILESEARCH_EMBEDDING_OFFLINE", None)
    environment.pop("FILESEARCH_ORT_PROFILE", None)
    environment.pop("FILESEARCH_ORT_INTER_THREADS", None)
    log = (output / "service.log").open("w", encoding="utf-8")
    process = None
    database = output / "data/search.db"

    def file_hash(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def cache_entries():
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            return dict(connection.execute("SELECT cache_key, used_at FROM passage_vector_cache"))

    def api(path, payload=None):
        request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    def start():
        nonlocal process
        process = subprocess.Popen([str(executable), "--listen", f"127.0.0.1:{port}",
                                    "--data-dir", str(output / "data"),
                                    "--model-dir", str(model_dir)],
                                   env=environment, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("Search core exited; inspect service.log")
            try:
                api("/health")
                return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("Search core did not start")

    def stop():
        nonlocal process
        if process is not None:
            process.terminate()
            process.wait(timeout=15)
            process = None

    def index(pause=False):
        assert api("/index", {"paths": [str(source)]})["accepted"]
        started = time.perf_counter()
        last_report = started
        samples = []
        paused = False
        retained = None
        peak_process_memory_bytes = 0
        pause_response_seconds = None
        while True:
            requested = time.perf_counter()
            stats = api("/stats")
            latency = time.perf_counter() - requested
            memory = psutil.Process(process.pid).memory_info()
            peak_process_memory_bytes = max(peak_process_memory_bytes, getattr(memory, "peak_wset", memory.rss))
            samples.append(dict(seconds=time.perf_counter() - started,
                                windows=stats.get("embedding_windows_processed", 0),
                                total=stats.get("embedding_windows_total", 0),
                                request_seconds=latency, status=stats["status"]))
            if pause and not paused and stats.get("embedding_windows_processed", 0) >= 4:
                pause_started = time.perf_counter()
                api("/index/pause", {})
                deadline = time.perf_counter() + 30
                while api("/stats")["status"] != "paused":
                    if time.perf_counter() > deadline:
                        raise TimeoutError("Index did not pause between inference batches")
                    time.sleep(0.1)
                pause_response_seconds = time.perf_counter() - pause_started
                before = api("/stats")["embedding_windows_processed"]
                time.sleep(0.3)
                assert api("/stats")["embedding_windows_processed"] == before
                api("/index/resume", {})
                paused = True
            if (pause and args.interrupt_after_windows > 0 and retained is None
                    and stats.get("embedding_windows_processed", 0) >= args.interrupt_after_windows):
                stop()
                retained = cache_entries()
                assert retained, "No inference batches persisted before interruption"
                with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                    assert connection.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL").fetchone()[0] == 0
                start()
            if stats["status"] == "ready":
                break
            if time.perf_counter() - last_report > 20:
                print(json.dumps(samples[-1]), flush=True)
                last_report = time.perf_counter()
            if time.perf_counter() - started > 7200:
                raise TimeoutError("Index timeout")
            time.sleep(0.25)
        failures = api("/failures")["items"]
        if failures:
            raise RuntimeError(f"Index failures: {failures}")
        assert stats["document_count"] == 1
        timing = api("/file-timings")["items"][0]
        if retained is not None:
            final_cache = cache_entries()
            assert all(final_cache.get(key) == used_at for key, used_at in retained.items()), "Completed windows were recomputed after restart"
        return dict(wall_seconds=time.perf_counter() - started, timing=timing,
                    max_stats_request_seconds=max(sample["request_seconds"] for sample in samples),
                    window_progress_updates=len({sample["windows"] for sample in samples if sample["total"]}),
                    pause_verified=paused, embedding_backend=stats["embedding_backend"],
                    peak_process_memory_bytes=peak_process_memory_bytes,
                    pause_response_seconds=pause_response_seconds,
                    interrupted_windows_reused=len(retained) if retained is not None else 0)

    try:
        start()
        cold = index(pause=True)
        stop()
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        start()
        warm = index()
        assert warm["timing"]["embedding_ms"] < cold["timing"]["embedding_ms"]
        result = dict(full_document=args.full, source_bytes=args.file.stat().st_size,
                      tested_bytes=target.stat().st_size, cold=cold, restart_cache=warm,
                      source_sha256=file_hash(args.file), tested_sha256=file_hash(target),
                      model_sha256=file_hash(model_dir / "model.onnx"),
                      executable_sha256=file_hash(executable))
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    finally:
        stop()
        log.close()


if __name__ == "__main__":
    main()
