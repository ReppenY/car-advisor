"""Recall and precision of passage retrieval. Does not change retrieval.py.

Hybrid uses search(). BM25 and dense use the same index and the same
diversify step, so one article cannot fill the whole list.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_advisor.paths import ROOT as DATA_ROOT  # noqa: E402
from car_advisor.retrieval import _diversify, _index, collection, embed, search, tokenize  # noqa: E402
from car_advisor.store import connect  # noqa: E402

LABELS = ROOT / "eval" / "retrieval_labels.json"
OUT = ROOT / "eval" / "retrieval_metrics.json"


def as_hit(row):
    return {
        "file": f"{row['slug']}.md",
        "start_line": row["start_line"],
        "end_line": row["end_line"],
    }


def ranked(query, k):
    conn = connect()
    try:
        index, rows = _index(conn, DATA_ROOT)
    finally:
        conn.close()
    by_id = {row["id"]: row for row in rows}
    scores = index.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    lexical = []
    for position in order:
        if scores[position] <= 0:
            break
        lexical.append(rows[position]["id"])
    col = collection()
    count = col.count()
    dense = []
    if count:
        found = col.query(query_embeddings=embed([query], query=True), n_results=min(k * 4, count))
        dense = [chunk_id for chunk_id in found["ids"][0] if chunk_id in by_id]
    return {
        "bm25": [as_hit(by_id[chunk_id]) for chunk_id in _diversify(lexical, by_id, k)],
        "dense": [as_hit(by_id[chunk_id]) for chunk_id in _diversify(dense, by_id, k)],
        "hybrid": search(query, k=k),
    }


def key(hit):
    return f"{hit['file']}:{hit['start_line']}-{hit['end_line']}"


def recall_at(hits, relevant, n):
    if not relevant:
        return 0.0
    return 1.0 if any(key(hit) in relevant for hit in hits[:n]) else 0.0


def mrr(hits, relevant):
    for index, hit in enumerate(hits, start=1):
        if key(hit) in relevant:
            return 1 / index
    return 0.0


def precision_at(hits, labels, k):
    if k == 0:
        return 0.0
    relevant = 0
    for hit in hits[:k]:
        label = labels.get(key(hit))
        if label and label["relevant"]:
            relevant += 1
    return relevant / k


def main():
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    k = data["k"]
    rows = []
    unlabeled = []
    for case in data["cases"]:
        relevant = {name for name, label in case["passages"].items() if label["relevant"]}
        found = ranked(case["query"], k)
        for method, hits in found.items():
            for hit in hits:
                if key(hit) not in case["passages"]:
                    unlabeled.append({"case": case["id"], "method": method, "passage": key(hit)})
            rows.append({
                "case": case["id"],
                "category": case["category"],
                "method": method,
                "hit_at_1": recall_at(hits, relevant, 1),
                "hit_at_3": recall_at(hits, relevant, 3),
                "hit_at_k": recall_at(hits, relevant, k),
                "mrr": mrr(hits, relevant),
                "precision_at_k": precision_at(hits, case["passages"], k),
                "returned": [key(hit) for hit in hits],
            })
    summary = aggregate(rows)
    payload = {"k": k, "rows": rows, "summary": summary, "unlabeled": unlabeled}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'method':<8} {'category':<14} {'hit@1':>6} {'hit@3':>6} {'hit@k':>6} {'MRR':>6} {'P@k':>6}")
    for row in summary:
        print(f"{row['method']:<8} {row['category']:<14} {row['hit_at_1']:>6.2f} {row['hit_at_3']:>6.2f} {row['hit_at_k']:>6.2f} {row['mrr']:>6.2f} {row['precision_at_k']:>6.2f}")
    if unlabeled:
        print("unlabeled passages:", unlabeled)
    return payload


def aggregate(rows):
    groups = {}
    for row in rows:
        for category in (row["category"], "all"):
            groups.setdefault((row["method"], category), []).append(row)
    out = []
    for (method, category), items in groups.items():
        out.append({
            "method": method,
            "category": category,
            "hit_at_1": mean(items, "hit_at_1"),
            "hit_at_3": mean(items, "hit_at_3"),
            "hit_at_k": mean(items, "hit_at_k"),
            "mrr": mean(items, "mrr"),
            "precision_at_k": mean(items, "precision_at_k"),
        })
    return out


def mean(items, field):
    return round(sum(item[field] for item in items) / len(items), 4) if items else 0.0


if __name__ == "__main__":
    main()
