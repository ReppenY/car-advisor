"""Precision and recall of role=subject against the saved fixture."""

from car_advisor.resolve import compact
from car_advisor.store import connect
from car_advisor.subjects import SUBJECTS


def rows():
    conn = connect()
    try:
        return conn.execute("SELECT slug, make, model, role FROM cars").fetchall()
    finally:
        conn.close()


def subject_sets(found=None):
    found = rows() if found is None else found
    got = {}
    for row in found:
        if row["role"] != "subject":
            continue
        got.setdefault(row["slug"], set()).add(compact(f"{row['make']} {row['model']}"))
    expected = {slug: {compact(name) for name in names} for slug, names in SUBJECTS.items()}
    return got, expected


def role_metrics(found=None):
    got, expected = subject_sets(found)
    files = sorted(set(got) | set(expected))
    mismatches = []
    true_positive = false_positive = false_negative = 0
    for slug in files:
        predicted = got.get(slug, set())
        gold = expected.get(slug, set())
        true_positive += len(predicted & gold)
        for name in sorted(predicted - gold):
            false_positive += 1
            mismatches.append({"file": f"{slug}.md", "kind": "false_subject", "family": name})
        for name in sorted(gold - predicted):
            false_negative += 1
            mismatches.append({"file": f"{slug}.md", "kind": "missed_subject", "family": name})
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "mismatches": mismatches,
    }
