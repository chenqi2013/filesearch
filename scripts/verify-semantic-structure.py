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


QUERIES = ["排行榜", "榜单", "排名", "大学排行榜", "院校名次", "钱款", "金额",
           "赔款", "逾期付款", "违约金", "现金", "欠钱", "付款", "天气预报",
           "红烧肉做法", "显卡驱动安装", "宠物饲养", "旅游景点", "工资", "家常菜",
           "劳动合同", "电影推荐", "服务器故障", "银行利息"]
DISTRACTORS = {
    "天气记录.txt": "城市天气预报\n今天多云转晴，气温18到25摄氏度，周末有小雨，出行记得携带雨伞。",
    "家常菜谱.txt": "家常红烧肉做法\n五花肉切块焯水，用冰糖炒糖色，加入生抽老抽和八角，小火炖至软烂。",
    "电脑维护.txt": "电脑显卡驱动安装说明\n从官方网站下载适用于操作系统的驱动，卸载旧版本后重新安装，重启电脑。",
    "宠物指南.txt": "宠物饲养常识\n猫狗需要定期接种疫苗，每日提供干净饮水，合理控制喂食量，注意卫生。",
    "旅游计划.txt": "旅游景点推荐\n杭州西湖、苏州园林与黄山风景区适合假期游览，提前查询交通与住宿。",
    "劳动合同.txt": "劳动合同\n员工每月工资在次月十日发放，试用期三个月，约定工作岗位、工作时间和社会保险。",
    "银行存款.txt": "银行存款业务\n定期存款到期后支付本金和利息，利息按本金、利率与存期计算，提前支取按活期利率计息。",
    "服务器运维.txt": "服务器故障处理\n服务无法连接时检查网络、防火墙、磁盘剩余空间和进程日志，确认端口是否监听。",
    "电影推荐.txt": "电影推荐\n喜剧和动画电影适合全家一起观看，温馨轻松的剧情适合周末家庭聚会。",
}
TARGETS = {
    **dict.fromkeys(["排行榜", "排名", "大学排行榜", "院校名次"], "QS 2026世界大学排名.xlsx"),
    **dict.fromkeys(["钱款", "金额", "赔款", "逾期付款", "违约金", "付款"], "2024-03-供应商合作协议.pdf"),
}
NEGATIVES = ["天气预报", "红烧肉做法", "显卡驱动安装", "宠物饲养", "旅游景点", "工资", "家常菜", "电影推荐", "服务器故障", "银行利息"]


def main():
    parser = argparse.ArgumentParser(description="Isolated real-file semantic recall regression; never changes the user's index")
    parser.add_argument("--file", type=Path, action="append", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline-executable", type=Path)
    parser.add_argument("--verify-upgrade", action="store_true")
    parser.add_argument("--novel-sample", type=Path, help="Adds 64 sampled passages only, not the full novel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.verify_upgrade and not args.baseline_executable:
        parser.error("--verify-upgrade requires --baseline-executable")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    sources = []
    for original in args.file:
        shutil.copy2(original, source / original.name)
        sources.append(dict(name=original.name, bytes=original.stat().st_size,
                            sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    for name, text in DISTRACTORS.items():
        (source / name).write_text(text, encoding="utf-8")
    if args.novel_sample:
        text = args.novel_sample.read_text(encoding="utf-8-sig")
        sample = "\n".join(text[index * (len(text) - 480) // 63:][:480] for index in range(64))
        (source / args.novel_sample.name).write_text(sample, encoding="utf-8")
    environment = dict(os.environ, ORT_DYLIB_PATH=str(root / "src-tauri/binaries/onnxruntime.dll"), RUST_LOG="info")
    environment.pop("FILESEARCH_EMBEDDING_OFFLINE", None)
    result = dict(sources=sources, novel_full_document_measured=False, runs={})
    configurations = [("baseline", args.baseline_executable), ("candidate", args.executable)]
    if args.verify_upgrade:
        configurations.append(("upgrade", args.executable))
    for label, executable in configurations:
        if executable is None:
            continue
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        data = output / label
        previous_cache = {}
        if label == "upgrade":
            shutil.copytree(output / "baseline", data)
            with sqlite3.connect(data / "search.db") as database:
                previous_cache = dict(database.execute("SELECT cache_key,used_at FROM passage_vector_cache"))
        else:
            data.mkdir()
        base = f"http://127.0.0.1:{port}"

        def api(path, payload=None):
            request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=30) as response:
                return json.load(response)

        with (data / "service.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(executable.resolve()), "--listen", f"127.0.0.1:{port}",
                                        "--data-dir", str(data), "--model-dir", str(root / "assets/models/embedding-rwkv-tiny")],
                                       stdout=log, stderr=log, env=environment,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError(f"{label} exited: {data / 'service.log'}")
                    try:
                        health = api("/health")
                        break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise TimeoutError(f"{label} startup timed out")
                if label != "upgrade":
                    assert api("/index", dict(paths=[str(source)]))["accepted"]
                started = time.perf_counter()
                while time.perf_counter() - started < 1200:
                    stats = api("/stats")
                    if stats["status"] == "ready":
                        break
                    time.sleep(.25)
                else:
                    raise TimeoutError(f"{label} indexing timed out")
                run = dict(health=health, stats=stats, index_seconds=time.perf_counter() - started,
                           timings=api("/file-timings?limit=100"), searches=[], filtered_searches=[])
                for query in QUERIES:
                    for mode in ("semantic", "hybrid", "keyword"):
                        started_query = time.perf_counter()
                        response = api("/search", dict(query=query, mode=mode, limit=20))
                        run["searches"].append(dict(query=query, mode=mode,
                            seconds=time.perf_counter() - started_query,
                            results=[dict(name=row["name"], score=row["score"], snippet=row["snippet"]) for row in response["results"]]))
                        if mode == "semantic":
                            print(json.dumps(dict(run=label, query=query,
                                results=[(row["name"], row["score"]) for row in response["results"]]), ensure_ascii=False), flush=True)
                if label != "baseline":
                    for query, extension in (("排行榜", "xlsx"), ("钱款", "pdf")):
                        response = api("/search", dict(query=query, mode="semantic", limit=1, extension=extension))
                        run["filtered_searches"].append(dict(query=query, extension=extension, results=response["results"]))
                        assert response["results"] and response["results"][0]["name"] == TARGETS[query]
                result["runs"][label] = run
            finally:
                process.terminate()
                process.wait(timeout=15)
        with sqlite3.connect(data / "search.db") as database:
            run["body_vectors"] = database.execute("SELECT COUNT(*) FROM semantic_passages").fetchone()[0]
            if database.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='semantic_anchors'").fetchone()[0]:
                run["anchors"] = database.execute("SELECT d.name, a.text, c.position FROM semantic_anchors a JOIN documents d ON d.id=a.document_id JOIN chunks c ON c.id=a.chunk_id ORDER BY d.name,a.id").fetchall()
            if label == "upgrade":
                current_cache = dict(database.execute("SELECT cache_key,used_at FROM passage_vector_cache"))
                profile = database.execute("SELECT value FROM settings WHERE key='embedding_profile'").fetchone()[0]
                active_keys = [hashlib.sha256((profile + '\0' + name + '\n' + text).encode()).hexdigest()
                    for name, text in database.execute("SELECT d.name,p.text FROM semantic_passages p JOIN documents d ON d.id=p.document_id")]
                run["reused_body_vectors"] = sum(key in previous_cache and current_cache.get(key) == previous_cache[key] for key in active_keys)
                assert run["reused_body_vectors"] > 0
                assert run["anchors"], "Existing index was not upgraded"
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for label in ("candidate", "upgrade"):
        if label not in result["runs"]:
            continue
        candidate = result["runs"][label]
        candidate["target_recall"] = {}
        for mode in ("semantic", "hybrid"):
            rows = [row for row in candidate["searches"] if row["mode"] == mode and row["query"] in TARGETS]
            passed = sum(bool(row["results"]) and row["results"][0]["name"] == TARGETS[row["query"]] for row in rows)
            candidate["target_recall"][mode] = dict(top1=passed, queries=len(rows))
            assert passed == len(rows), f"{label} {mode} target recall failed; inspect result.json"
            for row in candidate["searches"]:
                if row["mode"] == mode and row["query"] in NEGATIVES:
                    assert not any(hit["name"] in set(TARGETS.values()) for hit in row["results"]), row
        for query in ("金额", "逾期付款", "违约金", "付款"):
            row = next(row for row in candidate["searches"] if row["mode"] == "keyword" and row["query"] == query)
            assert any(hit["name"] == TARGETS[query] for hit in row["results"]), row
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
