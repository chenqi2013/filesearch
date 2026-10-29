import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sqlite3
import statistics
import subprocess
import time
from urllib.request import Request, urlopen

from importlib.util import module_from_spec, spec_from_file_location


def normalize(text):
    return "".join(text.lower().split())


def summarize(rows):
    ranks = [row["target_rank"] for row in rows]
    total = len(rows)
    return dict(total=total, top1=sum(rank == 1 for rank in ranks),
                hit3=sum(rank is not None and rank <= 3 for rank in ranks),
                hit10=sum(rank is not None and rank <= 10 for rank in ranks),
                mrr10=sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / total if total else None)


def main():
    parser = argparse.ArgumentParser(description="Frozen independent query evaluation; never modifies the source index")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path(__file__).parent / "fixtures/semantic-generalization-20261027.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    suite_bytes = args.suite.read_bytes()
    suite = json.loads(suite_bytes)
    spec = spec_from_file_location("ranking_verification", root / "scripts/verify-search-ranking.py")
    ranking = module_from_spec(spec)
    spec.loader.exec_module(ranking)
    source_path = args.database.resolve()
    source_before = ranking.vector_state(source_path)
    data = output / "data"
    data.mkdir()
    snapshot = data / "search.db"
    with sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(snapshot) as target:
            source.backup(target)
            target.execute("DELETE FROM directories")
    before = ranking.vector_state(snapshot)
    texts = defaultdict(str)
    with sqlite3.connect(snapshot) as database:
        for name, text in database.execute("SELECT d.name,c.text FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY d.id,c.position"):
            texts[name] += text + "\n"
    cases = []
    for kind in ("new_positive", "controls"):
        for group, query, evidence in suite[kind]:
            targets = suite["target_groups"][group]
            assert any(normalize(evidence) in normalize(texts[name]) for name in targets), (group, query, evidence)
            cases.append(dict(kind=kind, group=group, query=query, targets=targets, evidence=evidence,
                              full_query_absent=all(normalize(query) not in normalize(texts[name]) for name in targets)))
    for kind in ("negative", "ambiguous"):
        cases.extend(dict(kind=kind, group=kind, query=query, targets=[]) for query in suite[kind])
    assert len({case["query"] for case in cases}) == len(cases), "Duplicate queries"
    previous = set(ranking.QUERIES)
    assert not any(case["query"] in previous for case in cases if case["kind"] == "new_positive"), "New queries must not overlap tuning queries"
    report = dict(suite_sha256=hashlib.sha256(suite_bytes).hexdigest(),
                  executable=str(args.executable.resolve()), executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  before=before, cases=cases, searches=[], label_validation="all evidence quotes found in target indexed text",
                  evaluation_changes_search_code=False, full_novel_included=True)
    (output / "frozen-cases.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def api(path, payload=None):
        request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    environment = dict(os.environ, ORT_DYLIB_PATH=str(root / "src-tauri/binaries/onnxruntime.dll"), RUST_LOG="info")
    environment.pop("FILESEARCH_EMBEDDING_OFFLINE", None)
    with (data / "service.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(args.executable.resolve()), "--listen", f"127.0.0.1:{port}",
                                    "--data-dir", str(data), "--model-dir", str(root / "assets/models/embedding-rwkv-tiny")],
                                   stdout=log, stderr=log, env=environment,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for attempt in range(300):
                if process.poll() is not None:
                    raise RuntimeError(f"Evaluation service exited; see {data / 'service.log'}")
                try:
                    api("/health")
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise TimeoutError("Evaluation startup timed out")
            report["stats"] = api("/stats")
            assert report["stats"]["status"] == "ready", report["stats"]
            api("/search", dict(query="软件需求", mode="semantic", limit=10))
            order = list(cases)
            random.Random(20261027).shuffle(order)
            for index, case in enumerate(order):
                modes = ["semantic", "hybrid", "keyword"]
                modes = modes[index % 3:] + modes[:index % 3]
                for mode in modes:
                    started = time.perf_counter()
                    response = api("/search", dict(query=case["query"], mode=mode, limit=10))
                    elapsed = time.perf_counter() - started
                    results = response["results"]
                    rank = next((position for position, hit in enumerate(results, 1) if hit["name"] in case["targets"]), None)
                    row = dict(**case, mode=mode, seconds=elapsed, target_rank=rank,
                               evidence_in_target_snippet=any(normalize(case.get("evidence", "")) in normalize(hit["snippet"])
                                                             for hit in results if hit["name"] in case["targets"]),
                               results=[dict(name=hit["name"], score=hit["score"], snippet=hit["snippet"]) for hit in results])
                    report["searches"].append(row)
                if (index + 1) % 20 == 0:
                    print(f"Evaluated {index + 1}/{len(cases)} queries", flush=True)
            report["health"] = api("/health")
            assert report["health"]["embedding_backend"] == "cpu", report["health"]
        finally:
            process.terminate()
            process.wait(timeout=15)
    report["after"] = ranking.vector_state(snapshot)
    report["source_unchanged"] = source_before == ranking.vector_state(source_path)
    assert report["before"] == report["after"], "Evaluation changed stored vectors/cache"
    assert report["source_unchanged"], "Source index changed during evaluation; investigate concurrent activity"
    summary = {}
    for mode in ("semantic", "hybrid", "keyword"):
        positive = [row for row in report["searches"] if row["mode"] == mode and row["kind"] == "new_positive"]
        negatives = [row for row in report["searches"] if row["mode"] == mode and row["kind"] == "negative"]
        controls = [row for row in report["searches"] if row["mode"] == mode and row["kind"] == "controls"]
        keyword_misses = {row["query"] for row in report["searches"]
                          if row["mode"] == "keyword" and row["kind"] == "new_positive" and row["target_rank"] is None}
        durations = sorted(row["seconds"] for row in report["searches"] if row["mode"] == mode)
        summary[mode] = dict(new_positive=summarize(positive),
                             by_group={group: summarize([row for row in positive if row["group"] == group])
                                       for group in suite["target_groups"]},
                             full_query_absent=summarize([row for row in positive if row["full_query_absent"]]),
                             keyword_misses=summarize([row for row in positive if row["query"] in keyword_misses]),
                             controls=summarize(controls),
                             negatives=dict(total=len(negatives), empty=sum(not row["results"] for row in negatives),
                                            returned_queries=[row["query"] for row in negatives if row["results"]]),
                             latency_ms=dict(median=statistics.median(durations) * 1000,
                                             p95=durations[min(len(durations) - 1, int(len(durations) * .95))] * 1000))
    report["summary"] = summary
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        assert baseline["suite_sha256"] == report["suite_sha256"], "Comparison requires the same frozen suite"
        assert baseline["before"] == report["before"], "Comparison requires the same document/vector snapshot"
        previous = {(row["query"], row["mode"]): row for row in baseline["searches"]}
        comparison = dict(positive_targets_lost=[], positive_targets_gained=[],
                          negative_documents_added=[], negative_documents_removed=[], keyword_rankings_changed=[])
        for row in report["searches"]:
            before_row = previous[(row["query"], row["mode"])]
            if row["kind"] in ("new_positive", "controls"):
                if before_row["target_rank"] is not None and row["target_rank"] is None:
                    comparison["positive_targets_lost"].append(dict(query=row["query"], mode=row["mode"]))
                if before_row["target_rank"] is None and row["target_rank"] is not None:
                    comparison["positive_targets_gained"].append(dict(query=row["query"], mode=row["mode"]))
            if row["kind"] == "negative":
                prior = {hit["name"] for hit in before_row["results"]}
                current = {hit["name"] for hit in row["results"]}
                for label, names in (("negative_documents_added", current - prior),
                                     ("negative_documents_removed", prior - current)):
                    comparison[label].extend(dict(query=row["query"], mode=row["mode"], name=name) for name in sorted(names))
            if row["mode"] == "keyword":
                ranking = lambda item: [(hit["name"], hit["score"]) for hit in item["results"]]
                if ranking(before_row) != ranking(row):
                    comparison["keyword_rankings_changed"].append(row["query"])
        report["comparison"] = comparison
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 独立换词／口语查询测试明细", "", "本报告仅评估固定样本，不代表通用准确率。目标文件命中不等于准确定位答案段落。", "",
             "| 类别 | 查询 | 语义目标排名 | 混合目标排名 | 关键词目标排名 | 语义首位 |", "|---|---|---:|---:|---:|---|"]
    for case in cases:
        rows = {row["mode"]: row for row in report["searches"] if row["query"] == case["query"]}
        ranks = [str(rows[mode]["target_rank"] or "—") if case["targets"] else "不计目标排名"
                 for mode in ("semantic", "hybrid", "keyword")]
        top = rows["semantic"]["results"]
        lines.append(f"| {case['kind']}/{case['group']} | {case['query']} | {' | '.join(ranks)} | {top[0]['name'] if top else '无结果'} |")
    (output / "details.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if args.baseline:
        assert not comparison["positive_targets_lost"], comparison["positive_targets_lost"]
        assert not comparison["keyword_rankings_changed"], comparison["keyword_rankings_changed"]
        assert not comparison["negative_documents_added"], comparison["negative_documents_added"]


if __name__ == "__main__":
    main()
