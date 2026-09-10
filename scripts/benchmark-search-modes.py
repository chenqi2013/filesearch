from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen


def request(base_url, route, body=None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    query = Request(base_url + route, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(query, timeout=60) as response:
        return json.load(response)


def pages(base_url, route):
    items = []
    while True:
        page = request(base_url, f"/{route}?offset={len(items)}&limit=100")
        items.extend(page["items"])
        if len(items) >= page["total"]:
            return items
        if not page["items"]:
            raise RuntimeError("Unexpected empty page")


def fingerprint(snapshot):
    content = json.dumps(
        {key: snapshot[key] for key in ("documents", "chunks")},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def snapshot(base_url):
    stats = request(base_url, "/stats")
    if stats["status"] != "ready":
        raise RuntimeError("Wait for indexing to complete")
    return {"stats": stats, "documents": pages(base_url, "documents"),
            "chunks": pages(base_url, "chunks")}


def summarize(rows):
    positives = [row for row in rows if row["kind"] != "negative"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    result = {}
    if positives:
        count = len(positives)
        result.update({
            "queries": count,
            "hit@1": sum(row["rank"] == 1 for row in positives) / count,
            "hit@3": sum(row["rank"] is not None and row["rank"] <= 3 for row in positives) / count,
            "hit@5": sum(row["rank"] is not None and row["rank"] <= 5 for row in positives) / count,
            "mrr@10": sum(1 / row["rank"] if row["rank"] is not None and row["rank"] <= 10 else 0
                          for row in positives) / count,
            "median_server_ms": statistics.median(row["server_ms"] for row in positives),
            "mean_server_ms": statistics.mean(row["server_ms"] for row in positives),
            "max_server_ms": max(row["server_ms"] for row in positives),
        })
    if negatives:
        result["negative_queries"] = len(negatives)
        result["negative_empty_results"] = sum(row["returned"] == 0 for row in negatives)
    return result


def main():
    parser = argparse.ArgumentParser(description="Read-only evaluation of live desktop search modes")
    parser.add_argument("--base-url", default="http://127.0.0.1:47653")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = snapshot(args.base_url)
    output = {"created_at": datetime.now(timezone.utc).isoformat(),
              "corpus_sha256": fingerprint(before), **before}
    if args.cases:
        cases_bytes = args.cases.read_bytes()
        cases = json.loads(cases_bytes)
        if cases["corpus_sha256"] != fingerprint(before):
            raise RuntimeError("Cases refer to a different corpus")
        document_ids = {document["id"] for document in before["documents"]}
        queries = cases["queries"]
        if len({case["id"] for case in queries}) != len(queries):
            raise ValueError("Duplicate case IDs")
        for case in queries:
            expected = set(case["relevant_ids"])
            if not expected.issubset(document_ids):
                raise ValueError(f"Unknown document in case {case['id']}")
            if (case["kind"] == "negative") != (not expected):
                raise ValueError(f"Invalid relevance labels in case {case['id']}")
        modes = ("keyword", "semantic", "hybrid")
        for mode in modes:
            request(args.base_url, "/search", {"query": "初始化检索", "mode": mode, "limit": 50})
        rows = []
        for case_index, case in enumerate(queries):
            ordered_modes = modes[case_index % 3:] + modes[:case_index % 3]
            for mode in ordered_modes:
                started = time.perf_counter()
                response = request(args.base_url, "/search", {
                    "query": case["query"], "mode": mode, "limit": 50, "extension": None,
                })
                expected = set(case["relevant_ids"])
                rank = next((position for position, result in enumerate(response["results"], 1)
                             if result["id"] in expected), None)
                rows.append({**case, "mode": mode, "rank": rank,
                             "returned": len(response["results"]), "server_ms": response["elapsed_ms"],
                             "wall_ms": (time.perf_counter() - started) * 1000,
                             "results": response["results"]})
            print(f"Completed {case_index + 1}/{len(queries)}: {case['id']}", flush=True)
        after = snapshot(args.base_url)
        if fingerprint(before) != fingerprint(after) or any(
            before["stats"][key] != after["stats"][key]
            for key in ("last_indexed", "embedding_model", "embedding_backend")
        ):
            raise RuntimeError("Corpus or model changed during evaluation")
        output.update({"cases_sha256": hashlib.sha256(cases_bytes).hexdigest(), "rows": rows,
                       "summary": {mode: summarize([row for row in rows if row["mode"] == mode])
                                   for mode in modes},
                       "by_kind": {kind: {mode: summarize([row for row in rows if row["mode"] == mode
                                                          and row["kind"] == kind]) for mode in modes}
                                   for kind in sorted({case["kind"] for case in queries})}})
        print(json.dumps({"summary": output["summary"], "by_kind": output["by_kind"]}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
