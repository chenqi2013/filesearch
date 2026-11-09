"""Exercise crash recovery, retries, filtering and previews in an isolated real-RWKV index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for index in range(42):
        (source / f"record-{index:02}.txt").write_text(f"验收查询记录 资料编号{index:02}。这是第{index:02}份客户资料。", encoding="utf-8")
    novel = source / "long.txt"
    novel.write_text("\n".join(f"第{index}节档案：详细记载本地索引系统中文档处理、进度恢复、搜索定位与数据保存的过程。" for index in range(1100)), encoding="utf-8")
    (source / "empty.txt").touch()
    broken = source / "broken.docx"
    broken.write_bytes(b"invalid zip archive")
    sheet = source / "positions.xlsx"
    with ZipFile(sheet, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>岗位</t></is></c><c r="B1" t="inlineStr"><is><t>城市</t></is></c></row><row r="18"><c r="A18" t="inlineStr"><is><t>办公室文员</t></is></c><c r="B18" t="inlineStr"><is><t>重庆</t></is></c></row></sheetData></worksheet>')
    originals = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    environment = dict(os.environ, ORT_DYLIB_PATH=str(root / "src-tauri/binaries/onnxruntime.dll"), RUST_LOG="info")
    environment.pop("FILESEARCH_EMBEDDING_OFFLINE", None)
    process = None
    report = {}

    def api(path, payload=None):
        request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    def until(predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("service exited; inspect service.log")
            try:
                value = api("/stats")
                if predicate(value):
                    return value
            except OSError:
                pass
            time.sleep(.05)
        raise TimeoutError("feature acceptance timed out")

    def start(log):
        return subprocess.Popen([str(args.executable.resolve()), "--listen", f"127.0.0.1:{port}", "--data-dir", str(output / "data"), "--model-dir", str(root / "assets/models/embedding-rwkv-tiny")], stdout=log, stderr=log, env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    with (output / "service.log").open("w", encoding="utf-8") as log:
        try:
            process = start(log)
            until(lambda stats: stats["status"] == "ready")
            api("/index", {"paths": [str(source)]})
            partial = until(lambda stats: stats.get("current_file", "") == str(novel) and stats.get("embedding_windows_processed", 0) >= 4 and stats["embedding_windows_processed"] < stats["embedding_windows_total"])
            api("/index/pause", {})
            paused = until(lambda stats: stats["status"] == "paused")
            assert paused["document_count"] >= 42, paused
            # Searching while paused must not deadlock, and does not resume indexing.
            assert api("/search", {"query": "验收查询记录", "mode": "keyword"})["results"]
            process.terminate()
            process.wait(timeout=15)
            database = output / "data/search.db"
            with sqlite3.connect(database) as connection:
                ids_before = connection.execute("SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.name='long.txt' ORDER BY c.position").fetchall()
                saved_windows = connection.execute("SELECT COUNT(*) FROM semantic_passages p JOIN documents d ON d.id=p.document_id WHERE d.name='long.txt'").fetchone()[0]
                connection.execute("DELETE FROM passage_vector_cache")
            process = start(log)
            resumed_paused = until(lambda stats: stats["status"] == "paused")
            assert resumed_paused["document_count"] >= 42
            api("/index/resume", {})
            finished = until(lambda stats: stats["status"] == "ready", timeout=240)
            assert finished["failed_count"] == 1 and finished["skipped_count"] == 1, finished
            assert finished["semantic_pending_count"] == 0, finished
            assert finished["embedding_backend"] == "cpu", finished
            with sqlite3.connect(database) as connection:
                ids_after = connection.execute("SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.name='long.txt' ORDER BY c.position").fetchall()
                final_windows = connection.execute("SELECT COUNT(*) FROM semantic_passages p JOIN documents d ON d.id=p.document_id WHERE d.name='long.txt'").fetchone()[0]
            assert ids_before == ids_after and final_windows > saved_windows > 0
            payload = dict(query="验收查询记录", mode="keyword", limit=20, directory=str(source), min_size=1)
            first = api("/search", payload)
            second = api("/search", dict(payload, offset=20))
            assert first["total"] == second["total"] == 42
            assert not {row["id"] for row in first["results"]} & {row["id"] for row in second["results"]}
            phrase = api("/search", dict(payload, exact_phrase="资料编号41"))
            assert len(phrase["results"]) == 1 and phrase["results"][0]["name"] == "record-41.txt"
            assert not api("/search", dict(payload, exact_phrase="资料编号41", exclude_terms=["资料编号41"]))["results"]
            for mode in ("everyday", "full"):
                api("/settings/resources", {"mode": mode})
                assert api("/stats")["resource_mode"] == mode
            hit = api("/search", {"query": "办公室", "mode": "keyword"})["results"][0]
            preview = api(f"/document-content?document_id={hit['id']}&query=%E5%8A%9E%E5%85%AC%E5%AE%A4")
            assert any("第 18 行" in span["label"] for span in preview["locations"]), preview
            assert preview["passages"] and preview["chunks"]["total"] > 0
            with ZipFile(broken, "w") as archive:
                archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>重新解析后的验收合同。</w:t></w:r></w:p></w:body></w:document>')
            api("/index/retry", {"paths": [str(broken)]})
            retried = until(lambda stats: stats["status"] == "ready" and stats["failed_count"] == 0)
            assert api("/failures")["total"] == 0 and api("/failures?skipped=true")["total"] == 1
            assert all(hashlib.sha256(path.read_bytes()).hexdigest() == originals[path.name] for path in source.iterdir() if path != broken)
            report.update(partial=partial, paused=paused, restart_paused=resumed_paused, finished=finished, retried=retried,
                          saved_windows=saved_windows, final_windows=final_windows, chunk_ids_preserved=True, global_vector_cache_cleared_before_restart=True,
                          pagination_total=first["total"], preview_locations=preview["locations"], unrelated_originals_unchanged=True)
            (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"passed": True, "saved_windows": saved_windows, "final_windows": final_windows, "pagination_total": first["total"]}), flush=True)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
