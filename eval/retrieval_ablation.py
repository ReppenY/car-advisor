"""What BM25 and the embeddings each add. Does not change search().

Hybrid at dense weight 1 is the production fusion. Dense-only and BM25-only
use the same candidate pool and the same diversify step as search().
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_advisor.paths import ROOT as DATA_ROOT  # noqa: E402
from car_advisor.retrieval import (  # noqa: E402
    RRF_K,
    _diversify,
    _index,
    collection,
    embed,
    search,
    tokenize,
)
from car_advisor.store import connect  # noqa: E402

LABELS = ROOT / "eval" / "retrieval_labels.json"
OUT = ROOT / "eval" / "retrieval_ablation.json"
WEIGHTS = (1.0, 1.5, 2.0)


def _candidates(query, k, files=None, root=None):
    root = root or DATA_ROOT
    conn = connect(root)
    try:
        index, rows = _index(conn, root)
    finally:
        conn.close()
    by_id = {row["id"]: row for row in rows}
    if not rows:
        return [], [], by_id
    slugs = {name.removesuffix(".md") for name in files} if files else None
    pool = max(k * 5, 20)

    dense_ids = []
    col = collection(root)
    count = col.count()
    if count:
        where = None
        if slugs:
            wanted = list(slugs)
            where = {"slug": wanted[0]} if len(wanted) == 1 else {"slug": {"$in": wanted}}
        found = col.query(
            query_embeddings=embed([query], query=True),
            n_results=min(pool, count),
            where=where,
        )
        dense_ids = [chunk_id for chunk_id in found["ids"][0] if chunk_id in by_id]

    lexical_ids = []
    if index is not None:
        scores = index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        for position in ranked:
            if scores[position] <= 0:
                break
            row = rows[position]
            if slugs and row["slug"] not in slugs:
                continue
            lexical_ids.append(row["id"])
            if len(lexical_ids) >= pool:
                break
    return dense_ids, lexical_ids, by_id


def _fuse(dense_ids, lexical_ids, dense_weight):
    scores = {}
    for ranking, weight in ((dense_ids, dense_weight), (lexical_ids, 1.0)):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank + 1)
    return [chunk_id for chunk_id, _ in sorted(scores.items(), key=lambda item: -item[1])]


def _hits(ordered, by_id, k):
    hits = []
    for chunk_id in _diversify(ordered, by_id, k):
        row = by_id[chunk_id]
        hits.append({
            "file": f"{row['slug']}.md",
            "slug": row["slug"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "heading": row["heading"],
            "text": row["text"],
        })
    return hits


def search_with(query, *, k=4, files=None, root=None, method="hybrid", dense_weight=1.0):
    """Same pool as search(). method is hybrid, dense, or bm25. Default hybrid weight is 1."""
    dense_ids, lexical_ids, by_id = _candidates(query, k, files, root)
    if method == "dense":
        ordered = dense_ids
    elif method == "bm25":
        ordered = lexical_ids
    else:
        ordered = _fuse(dense_ids, lexical_ids, dense_weight)
    return _hits(ordered, by_id, k)


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


def mean(items, field):
    return round(sum(item[field] for item in items) / len(items), 4) if items else 0.0


def methods_for(query, k):
    dense_ids, lexical_ids, by_id = _candidates(query, k)
    found = {
        "bm25": _hits(lexical_ids, by_id, k),
        "dense": _hits(dense_ids, by_id, k),
        "hybrid": _hits(_fuse(dense_ids, lexical_ids, 1.0), by_id, k),
    }
    for weight in WEIGHTS:
        found[f"rrf-{weight:g}"] = _hits(_fuse(dense_ids, lexical_ids, weight), by_id, k)
    return found


def score_case(case, k, found):
    relevant = {name for name, label in case["passages"].items() if label["relevant"]}
    rows = []
    unlabeled = []
    for method, hits in found.items():
        for hit in hits:
            if key(hit) not in case["passages"]:
                unlabeled.append({
                    "case": case["id"],
                    "method": method,
                    "passage": key(hit),
                    "text": hit["text"][:240],
                })
        rows.append({
            "case": case["id"],
            "category": case["category"],
            "query": case["query"],
            "method": method,
            "hit_at_1": recall_at(hits, relevant, 1),
            "hit_at_k": recall_at(hits, relevant, k),
            "mrr": round(mrr(hits, relevant), 4),
            "precision_at_k": precision_at(hits, case["passages"], k),
            "returned": [key(hit) for hit in hits],
        })
    return rows, unlabeled


def aggregate(rows):
    groups = {}
    for row in rows:
        for category in (row["category"], "all"):
            groups.setdefault((row["method"], category), []).append(row)
    out = []
    for (method, category), items in sorted(groups.items()):
        out.append({
            "method": method,
            "category": category,
            "n": len(items),
            "hit_at_1": mean(items, "hit_at_1"),
            "hit_at_k": mean(items, "hit_at_k"),
            "mrr": mean(items, "mrr"),
            "precision_at_k": mean(items, "precision_at_k"),
        })
    return out


def by_method(summary, method):
    return {row["category"]: row for row in summary if row["method"] == method}


def contribution(summary):
    hybrid = by_method(summary, "hybrid")
    dense = by_method(summary, "dense")
    bm25 = by_method(summary, "bm25")
    rows = []
    for category, base in hybrid.items():
        rows.append({
            "category": category,
            "hybrid_hit_at_1": base["hit_at_1"],
            "dense_hit_at_1": dense[category]["hit_at_1"],
            "bm25_hit_at_1": bm25[category]["hit_at_1"],
            "bm25_adds_hit_at_1": round(base["hit_at_1"] - dense[category]["hit_at_1"], 4),
            "embeddings_add_hit_at_1": round(base["hit_at_1"] - bm25[category]["hit_at_1"], 4),
            "hybrid_mrr": base["mrr"],
            "dense_mrr": dense[category]["mrr"],
            "bm25_mrr": bm25[category]["mrr"],
            "bm25_adds_mrr": round(base["mrr"] - dense[category]["mrr"], 4),
            "embeddings_add_mrr": round(base["mrr"] - bm25[category]["mrr"], 4),
            "hybrid_p_at_k": base["precision_at_k"],
            "dense_p_at_k": dense[category]["precision_at_k"],
            "bm25_p_at_k": bm25[category]["precision_at_k"],
        })
    return rows


def rescues_and_drags(rows):
    """Right means hit@1. A rescue is hybrid right where one method is wrong.
    A drag is hybrid wrong where the better single method is right.
    """
    grouped = {}
    for row in rows:
        if row["method"] not in ("hybrid", "dense", "bm25"):
            continue
        grouped.setdefault(row["case"], {})[row["method"]] = row
    rescues, drags = [], []
    for case, methods in grouped.items():
        hybrid = methods["hybrid"]["hit_at_1"]
        singles = {name: methods[name]["hit_at_1"] for name in ("dense", "bm25")}
        wrong = [name for name, hit in singles.items() if hit == 0]
        right = [name for name, hit in singles.items() if hit == 1]
        sample = methods["hybrid"]
        if hybrid == 1 and wrong:
            rescues.append({
                "case": case,
                "category": sample["category"],
                "query": sample["query"],
                "saved_from": wrong,
                "hybrid": methods["hybrid"]["returned"][0],
            })
        if hybrid == 0 and right:
            drags.append({
                "case": case,
                "category": sample["category"],
                "query": sample["query"],
                "better": right,
                "hybrid": methods["hybrid"]["returned"][0] if methods["hybrid"]["returned"] else "",
            })
    return rescues, drags


def check_matches_production(cases, k):
    mismatches = []
    for case in cases:
        produced = [key(hit) for hit in search(case["query"], k=k)]
        dense_ids, lexical_ids, by_id = _candidates(case["query"], k)
        fused = [key(hit) for hit in _hits(_fuse(dense_ids, lexical_ids, 1.0), by_id, k)]
        if fused != produced:
            mismatches.append({"case": case["id"], "search": produced, "fused": fused})
    return mismatches


def main():
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    k = data["k"]
    mismatches = check_matches_production(data["cases"], k)
    rows, unlabeled = [], []
    for case in data["cases"]:
        found = methods_for(case["query"], k)
        case_rows, missing = score_case(case, k, found)
        rows.extend(case_rows)
        unlabeled.extend(missing)
    summary = aggregate(rows)
    added = contribution(summary)
    rescues, drags = rescues_and_drags(rows)
    payload = {
        "k": k,
        "production_match": not mismatches,
        "mismatches": mismatches,
        "summary": summary,
        "contribution": added,
        "rescues": rescues,
        "drags": drags,
        "rows": rows,
        "unlabeled": unlabeled,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"production fusion matches search(): {not mismatches}")
    if unlabeled:
        seen = []
        for item in unlabeled:
            mark = (item["case"], item["passage"])
            if mark in seen:
                continue
            seen.append(mark)
            print(f"UNLABELED {item['case']} {item['passage']}")
            print(item["text"].replace("\n", " ")[:180])
    print(f"{'method':<10} {'category':<14} {'hit@1':>6} {'hit@k':>6} {'MRR':>6} {'P@k':>6}")
    for row in summary:
        if row["method"] in ("hybrid", "dense", "bm25") or row["category"] == "all":
            print(
                f"{row['method']:<10} {row['category']:<14} "
                f"{row['hit_at_1']:>6.2f} {row['hit_at_k']:>6.2f} {row['mrr']:>6.2f} {row['precision_at_k']:>6.2f}"
            )
    return payload


if __name__ == "__main__":
    main()
