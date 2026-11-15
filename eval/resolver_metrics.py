"""Resolver precision and recall. No model calls.

A confident prediction is exactly one family. Precision is confident predictions
that match the label, over all confident predictions. Recall is those matches
over labels that name one family. Returning candidates, or none, is not a
confident match. Style is labeled candidates because that alias is shared.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_advisor.resolve import allowed_distance, compact, damerau_levenshtein, match_aliases  # noqa: E402
from car_advisor.store import connect  # noqa: E402

NAMES = ROOT / "eval" / "resolver_names.json"
OUT = ROOT / "eval" / "resolver_metrics.json"


def aliases():
    conn = connect()
    try:
        return [
            (row["key"], row["family_id"], row["kind"])
            for row in conn.execute("SELECT key, family_id, kind FROM aliases")
        ]
    finally:
        conn.close()


def predict(query, rows, limit):
    key = compact(query)
    if not key:
        return "none", []
    exact = []
    for alias, family, _kind in rows:
        if alias == key and family not in exact:
            exact.append(family)
    if len(exact) == 1:
        return "match", exact
    if len(exact) > 1:
        return "candidates", exact
    if limit == 0:
        return "none", []
    hits = []
    for alias, family, _kind in rows:
        if damerau_levenshtein(key, alias) <= limit and family not in hits:
            hits.append(family)
    if len(hits) == 1:
        return "match", hits
    if len(hits) > 1:
        return "candidates", hits
    return "none", []


def predict_scaled(query, rows):
    found = match_aliases(query, rows)
    if found.status == "one":
        return "match", list(found.family_ids)
    if found.status == "candidates":
        return "candidates", list(found.family_ids)
    return "none", []


def score(items, rows, rule):
    limit = None if rule == "scaled" else int(rule)
    tp = fp = fn = 0
    wrong = []
    for item in items:
        gold = item["expect"]
        status, families = predict_scaled(item["name"], rows) if limit is None else predict(item["name"], rows, limit)
        predicted = families[0] if status == "match" else None
        if gold == "candidates":
            wanted = set(item.get("families") or [])
            if status == "match":
                fp += 1
                wrong.append({**item, "predicted": predicted, "why": "guessed one family for a shared alias"})
            elif status != "candidates" or (wanted and set(families) != wanted):
                wrong.append({
                    **item,
                    "predicted": families or status,
                    "why": "did not return both families",
                    "confident": False,
                })
            continue
        if gold == "none":
            if status == "match":
                fp += 1
                wrong.append({**item, "predicted": predicted, "why": "confident match for a name that should not resolve", "confident": True})
            elif status == "candidates":
                wrong.append({
                    **item,
                    "predicted": families,
                    "why": "returned candidates for a name that should not resolve",
                    "confident": False,
                })
            continue
        if status == "match" and predicted == gold:
            tp += 1
        elif status == "match":
            fp += 1
            fn += 1
            wrong.append({**item, "predicted": predicted, "why": "confident match to the wrong family", "confident": True})
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "rule": "length-scaled" if limit is None else f"distance {limit}",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "wrong_confident": [item for item in wrong if item.get("confident", True)],
        "other": [item for item in wrong if not item.get("confident", True)],
    }


def main():
    items = json.loads(NAMES.read_text(encoding="utf-8"))
    rows = aliases()
    rules = []
    for limit in (0, 1, 2):
        rules.append(score(items, rows, limit))
    rules.append(score(items, rows, "scaled"))
    payload = {"names": len(items), "rules": rules}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'rule':<16} {'precision':>10} {'recall':>8} {'wrong confident':>16}")
    for row in rules:
        print(f"{row['rule']:<16} {row['precision']:>10.3f} {row['recall']:>8.3f} {len(row['wrong_confident']):>16}")
        for miss in row["wrong_confident"]:
            print(f"    {miss['name']!r} -> {miss['predicted']} (expected {miss['expect']})")
    return payload


if __name__ == "__main__":
    main()
