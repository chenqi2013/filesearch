"""Build a deterministic NanoBEIR subset, retaining all positive judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", default="SciFact")
    parser.add_argument("--distractors", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.distractors < 0:
        parser.error("distractors must be nonnegative")
    rows, hashes = {}, {}
    for kind in ["corpus", "queries", "qrels"]:
        path = args.data / kind / f"Nano{args.name}-00000-of-00001.parquet"
        rows[kind] = parquet.read_table(path).to_pylist()
        with path.open("rb") as stream:
            hashes[kind] = hashlib.file_digest(stream, "sha256").hexdigest()
    positives = {}
    for row in rows["qrels"]:
        if row.get("score", 1) > 0:
            positives.setdefault(row["query-id"], set()).add(row["corpus-id"])
    selected = set().union(*(positives[row["_id"]] for row in rows["queries"]))
    negatives = [row for row in rows["corpus"] if row["_id"] not in selected]
    count = min(args.distractors, len(negatives))
    selected.update(negatives[index * (len(negatives) - 1) // max(1, count - 1)]["_id"]
                    for index in range(count))
    documents = [{"id": row["_id"], "text": (row.get("title", "") + " " + row["text"]).strip()}
                 for row in rows["corpus"] if row["_id"] in selected]
    assert len(documents) == len(selected), "Missing judged documents"
    result = dict(source=f"NanoBEIR-en/{args.name}", subset=True, source_sha256=hashes,
                  original_documents=len(rows["corpus"]),
                  documents=documents,
                  queries=[{"text": row["text"], "relevant": sorted(positives[row["_id"]])}
                           for row in rows["queries"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved subset: {len(documents)} documents, {len(result['queries'])} queries")


if __name__ == "__main__":
    main()
