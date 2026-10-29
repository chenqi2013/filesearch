"""Compare fixed retrieval runs, including snippets, without touching any index."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(before, after, queries=None):
    assert before["suite_sha256"] == after["suite_sha256"]
    assert before["before"] == after["before"] == after["after"]
    assert after["source_unchanged"]
    previous = {(row["query"], row["mode"]): row for row in before["searches"]}
    report = {}
    for mode in ("semantic", "hybrid", "keyword"):
        rows = [row for row in after["searches"] if row["mode"] == mode
                and (queries is None or row["query"] in queries)]
        positives = [row for row in rows if row["targets"]]
        negatives = [row for row in rows if row["kind"] == "negative"]
        changes = []
        snippet_gains, snippet_losses = [], []
        for row in rows:
            old = previous[row["query"], mode]
            identity = lambda value: [(hit["name"], hit["score"]) for hit in value["results"]]
            if identity(old) != identity(row):
                changes.append(row["query"])
            if row["targets"] and old["evidence_in_target_snippet"] != row["evidence_in_target_snippet"]:
                (snippet_gains if row["evidence_in_target_snippet"] else snippet_losses).append(row["query"])
            if mode == "keyword":
                assert old["results"] == row["results"], row["query"]
        report[mode] = dict(
            positive=len(positives), top1=sum(row["target_rank"] == 1 for row in positives),
            hit10=sum(row["target_rank"] is not None for row in positives),
            quoted_evidence_before=sum(previous[row["query"], mode]["evidence_in_target_snippet"] for row in positives),
            quoted_evidence_after=sum(row["evidence_in_target_snippet"] for row in positives),
            negative=len(negatives), negative_empty=sum(not row["results"] for row in negatives),
            ranking_or_score_changes=changes, snippet_gains=snippet_gains, snippet_losses=snippet_losses,
            source_unchanged=after["source_unchanged"],
        )
        assert not changes, (mode, changes)
        assert not snippet_losses, (mode, snippet_losses)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-prefix", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    previous = root / ".codex-tmp/search-evidence-20261028"
    prefix = args.candidate_prefix
    pairs = {
        "real_13_documents": (output / "trace-main/result.json", output / f"{prefix}final-main/result.json"),
        "new_24_positive_12_negative": (output / "new-baseline/result.json", output / f"{prefix}final-new/result.json"),
        "full_corpus_main": (previous / "verified-main/result.json", output / f"{prefix}full-main/result.json"),
        "full_corpus_supplement": (previous / "verified-holdout/result.json", output / f"{prefix}full-holdout/result.json"),
    }
    report = {name: compare(read(before), read(after)) for name, (before, after) in pairs.items()}
    fixture = read(root / "scripts/fixtures/semantic-refinement-20261029.json")
    report["reserved_new_12_positive"] = compare(
        read(output / "new-baseline/result.json"), read(output / f"{prefix}final-new/result.json"),
        {row[1] for row in fixture["new_positive"][12:]},
    )
    controls = read(output / f"{prefix}full-controls/result.json")
    assert not controls["regressions"]
    baseline, candidate = controls["runs"]["baseline"], controls["runs"]["candidate"]
    for old, new in zip(baseline["searches"], candidate["searches"]):
        assert (old["query"], old["mode"]) == (new["query"], new["mode"])
        assert [(hit["name"], hit["score"]) for hit in old["results"]] == [
            (hit["name"], hit["score"]) for hit in new["results"]], new["query"]
        if new["mode"] == "keyword":
            assert old["results"] == new["results"], new["query"]
    report["legacy_57_queries"] = dict(
        targets=candidate["targets"], rankings_and_scores_identical=True,
        index_unchanged=candidate["before"] == candidate["after"],
    )
    (output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
