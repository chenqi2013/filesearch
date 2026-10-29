import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import statistics
import subprocess
import time
from urllib.request import Request, urlopen


QUERIES = [
    "排行榜", "钱款", "榜单", "院校名次", "高校排名", "全球大学排名", "学校排名表",
    "有哪些世界名校", "排名靠前的大学", "货款", "结账", "支付款项", "拖欠货款",
    "没按时付款怎么办", "供应商交付不合格怎么办", "采购价格", "现金", "银行利息",
    "天气预报", "显卡驱动安装", "金额", "赔款", "逾期付款", "违约金", "付款",
    "红烧肉做法", "服务器故障", "办公室", "坐办公室的岗位", "以前很火的老歌",
    "适合全家一起看的轻松片子", "龙宠", "杨天", "黑洞", "麻省理工学院",
    "Praesent pu", "Praesent pulv", "Praesent pulvinar", "心率算法", "软件需求",
    "索引失败", "文件搜索", "数据备份", "大学排名", "世界高校排名", "高校名次",
    "大学名单", "世界一流大学", "供应商协议", "付款期限", "质量问题怎么赔偿",
    "采购合同", "违约责任", "医学影像诊断", "汽车维修手册", "蛋糕烘焙教程",
    "银河系",
]
UNIVERSITY = "QS 2026世界大学排名.xlsx"
CONTRACT = "2024-03-供应商合作协议.pdf"
NOVEL = "小说——超级龙宠.txt"
TARGETS = {
    **dict.fromkeys(["排行榜", "院校名次", "高校排名", "全球大学排名", "学校排名表",
                     "有哪些世界名校", "排名靠前的大学", "麻省理工学院", "大学排名",
                     "榜单", "世界高校排名", "高校名次", "大学名单", "世界一流大学"], UNIVERSITY),
    **dict.fromkeys(["钱款", "货款", "结账", "支付款项", "拖欠货款", "没按时付款怎么办",
                     "供应商交付不合格怎么办", "采购价格", "金额", "赔款", "逾期付款",
                     "违约金", "付款", "供应商协议", "付款期限", "质量问题怎么赔偿",
                     "采购合同", "违约责任"], CONTRACT),
    **dict.fromkeys(["Praesent pu", "Praesent pulv", "Praesent pulvinar"], "11.xlsx"),
    **dict.fromkeys(["龙宠", "杨天", "黑洞", "现金"], NOVEL),
}


def vector_state(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as database:
        digest = hashlib.sha256()
        counts = {}
        for table in ("documents", "semantic_passages", "semantic_anchors", "passage_vector_cache"):
            rows = database.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            counts[table] = len(rows)
            for row in rows:
                for value in row:
                    digest.update(repr(value).encode("utf-8"))
        counts["fingerprint"] = digest.hexdigest()
        return counts


def main():
    parser = argparse.ArgumentParser(description="Read-only source snapshot, full-corpus ranking regression")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--baseline-executable", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, ORT_DYLIB_PATH=str(root / "src-tauri/binaries/onnxruntime.dll"), RUST_LOG="info")
    environment.pop("FILESEARCH_EMBEDDING_OFFLINE", None)
    report = dict(queries=len(QUERIES), full_corpus=True, runs={})
    for label, executable in (("baseline", args.baseline_executable), ("candidate", args.executable)):
        data = output / label
        data.mkdir()
        database_path = data / "search.db"
        with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(database_path) as target:
                source.backup(target)
                target.execute("DELETE FROM directories")
        before = vector_state(database_path)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"

        def api(path, payload=None):
            request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)

        with (data / "service.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(executable.resolve()), "--listen", f"127.0.0.1:{port}",
                                        "--data-dir", str(data), "--model-dir", str(root / "assets/models/embedding-rwkv-tiny")],
                                       stdout=log, stderr=log, env=environment,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                for attempt in range(300):
                    if process.poll() is not None:
                        raise RuntimeError(f"{label} exited; see {data / 'service.log'}")
                    try:
                        health = api("/health")
                        break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise TimeoutError(f"{label} startup timed out")
                stats = api("/stats")
                assert stats["status"] == "ready", stats
                run = dict(health=health, stats=stats, before=before, searches=[], filtered_searches=[])
                api("/search", dict(query="软件需求", mode="semantic", limit=20))
                for query in QUERIES:
                    for mode in ("semantic", "hybrid", "keyword"):
                        started = time.perf_counter()
                        response = api("/search", dict(query=query, mode=mode, limit=20))
                        run["searches"].append(dict(query=query, mode=mode, seconds=time.perf_counter() - started,
                            results=[dict(name=row["name"], score=row["score"], snippet=row["snippet"])
                                     for row in response["results"]]))
                for query, extension, expected in (("排行榜", "xlsx", UNIVERSITY), ("钱款", "pdf", CONTRACT)):
                    rows = api("/search", dict(query=query, mode="semantic", extension=extension, limit=1))["results"]
                    run["filtered_searches"].append(dict(query=query, extension=extension, results=rows))
                run["health"] = api("/health")
                assert run["health"]["embedding_backend"] == "cpu", run["health"]
            finally:
                process.terminate()
                process.wait(timeout=15)
        run["after"] = vector_state(database_path)
        assert run["before"] == run["after"], "Searching must not alter documents or embedding vectors/cache"
        run["targets"] = {}
        for mode in ("semantic", "hybrid", "keyword"):
            rows = [row for row in run["searches"] if row["mode"] == mode and row["query"] in TARGETS]
            ranks = {row["query"]: next((rank for rank, hit in enumerate(row["results"], 1)
                                          if hit["name"] == TARGETS[row["query"]]), None) for row in rows}
            run["targets"][mode] = dict(ranks=ranks, top1=sum(rank == 1 for rank in ranks.values()),
                                         recall=sum(rank is not None for rank in ranks.values()), total=len(ranks))
            durations = [row["seconds"] for row in run["searches"] if row["mode"] == mode and row["query"].strip()]
            run.setdefault("latency", {})[mode] = dict(median=statistics.median(durations), maximum=max(durations))
        report["runs"][label] = run
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(run=label, targets=run["targets"], latency=run["latency"]), ensure_ascii=True), flush=True)
    baseline = report["runs"]["baseline"]
    candidate = report["runs"]["candidate"]
    report["regressions"] = [dict(mode=mode, query=query) for mode in ("semantic", "hybrid", "keyword")
                             for query, rank in baseline["targets"][mode]["ranks"].items()
                             if rank is not None and candidate["targets"][mode]["ranks"][query] is None]
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert not report["regressions"], report["regressions"]
    for row in candidate["searches"]:
        if row["mode"] == "keyword":
            previous = next(item for item in baseline["searches"]
                            if item["mode"] == row["mode"] and item["query"] == row["query"])
            assert [(hit["name"], hit["score"]) for hit in row["results"]] == [
                (hit["name"], hit["score"]) for hit in previous["results"]], row
        elif row["query"] == "排行榜":
            assert row["results"][0]["name"] == UNIVERSITY, row
            assert not any(hit["name"] == "11.xlsx" for hit in row["results"]), row
            assert any(hit["name"] == NOVEL for hit in row["results"]), row
        elif row["query"] == "显卡驱动安装":
            assert not row["results"], row
    for row in candidate["filtered_searches"]:
        expected = TARGETS[row["query"]]
        assert row["results"] and row["results"][0]["name"] == expected, row


if __name__ == "__main__":
    main()
