"""One table across the metric layers. Reads the saved runs. Does not call the agent."""

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from role_metrics import role_metrics  # noqa: E402

from answer_metrics import score_runs  # noqa: E402
from gate_metrics import rates, score_run  # noqa: E402
from resolver_metrics import main as resolver_main  # noqa: E402
from retrieval_metrics import main as retrieval_main  # noqa: E402

RESULTS = ROOT / "eval" / "results.json"
LABELS = ROOT / "eval" / "gate_labels.json"
OUT = ROOT / "eval" / "summary.json"


def variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return round(statistics.pvariance(values), 4)


def by_category(rows: list[dict], field: str) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row[field])
    return {
        category: {"mean": round(sum(values) / len(values), 4), "variance": variance(values)}
        for category, values in groups.items()
        if values
    }


def gate_rows():
    from car_advisor.store import connect

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    conn = connect()
    rows = []
    categories = {}
    try:
        for case in report["cases"]:
            categories[case["id"]] = case["category"]
            for run in case["runs"]:
                row = score_run(conn, case["id"], run, labels[case["id"]])
                row["category"] = case["category"]
                rows.append(row)
    finally:
        conn.close()
    return rows, report


def binary(row: dict) -> float | None:
    if not row["counted"]:
        return None
    return 0.0 if row["false_allow"] or row["false_block"] else 1.0


def main():
    resolver = resolver_main()
    roles = role_metrics()
    retrieval = retrieval_main()
    rows, report = gate_rows()
    answers = score_runs()
    per_case_gate = {}
    for row in rows:
        per_case_gate.setdefault(row["case"], []).append(binary(row))
    gate_values = [value for values in per_case_gate.values() for value in values if value is not None]
    answer_cases = {}
    for item in answers["runs"]:
        answer_cases.setdefault(item["case"], []).append(item)
    number_values = []
    citation_values = []
    list_precision = []
    list_recall = []
    for items in answer_cases.values():
        number_values.extend(1.0 if not item["number_problems"] else 0.0 for item in items)
        citation_values.extend(1.0 if not item["bad_citations"] else 0.0 for item in items)
        for item in items:
            if item["car_list"]:
                if item["car_list"]["precision"] is not None:
                    list_precision.append(item["car_list"]["precision"])
                list_recall.append(item["car_list"]["recall"])
    judge_values = [
        item["judge"]["claim_precision"]
        for item in answers["runs"]
        if item.get("judge") and item["judge"]["claim_precision"] is not None
    ]
    fact_values = [
        item["judge"]["fact_recall"]
        for item in answers["runs"]
        if item.get("judge") and item["judge"]["fact_recall"] is not None
    ]
    stamped = datetime.fromtimestamp(RESULTS.stat().st_mtime).isoformat(timespec="seconds")
    table = [
        {"layer": "resolver", "metric": "precision / recall by distance rule", "value": [
            {"rule": row["rule"], "precision": row["precision"], "recall": row["recall"], "wrong_confident": len(row["wrong_confident"])}
            for row in resolver["rules"]
        ]},
        {"layer": "roles", "metric": "precision / recall", "value": {"precision": roles["precision"], "recall": roles["recall"], "mismatches": roles["mismatches"]}},
        {"layer": "retrieval", "metric": "recall@k, MRR, precision@k", "value": retrieval["summary"]},
        {"layer": "gate", "metric": "precision / recall, false allows / false blocks", "value": rates(rows)},
        {"layer": "car lists", "metric": "precision / recall", "value": {
            "precision": round(sum(list_precision) / len(list_precision), 4) if list_precision else None,
            "recall": round(sum(list_recall) / len(list_recall), 4) if list_recall else None,
        }},
        {"layer": "answers", "metric": "misattributed numbers, bad citations", "value": {
            "runs_without_number_problems": round(sum(number_values) / len(number_values), 4) if number_values else None,
            "runs_without_bad_citations": round(sum(citation_values) / len(citation_values), 4) if citation_values else None,
            "number_problems": sum(len(item["number_problems"]) for item in answers["runs"]),
            "bad_citations": sum(len(item["bad_citations"]) for item in answers["runs"]),
            "claim_precision": round(sum(judge_values) / len(judge_values), 4) if judge_values else None,
            "fact_recall": round(sum(fact_values) / len(fact_values), 4) if fact_values else None,
            "judge": "provisional" if judge_values else "not run",
        }},
    ]
    payload = {
        "results_written": stamped,
        "runs_per_case": report.get("runs_per_case"),
        "cases": len(report["cases"]),
        "table": table,
        "variance": {
            "gate_decision": variance([value for value in gate_values]),
            "clean_numbers": variance(number_values),
            "clean_citations": variance(citation_values),
            "claim_precision": variance(judge_values),
            "fact_recall": variance(fact_values),
        },
        "by_category": {
            "gate": by_category(
                [{"category": row["category"], "score": binary(row)} for row in rows if binary(row) is not None],
                "score",
            ),
            "numbers": by_category(
                [{"category": item["category"], "score": 0.0 if item["number_problems"] else 1.0} for item in answers["runs"]],
                "score",
            ),
        },
        "resolver": resolver,
        "roles": roles,
        "gate_rows": rows,
        "answers": answers,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nresults {stamped}  runs/case={report.get('runs_per_case')}  cases={len(report['cases'])}")
    print(f"{'layer':<12} metric")
    for row in table:
        print(f"{row['layer']:<12} {row['metric']}")
        print(f"             {json.dumps(row['value'], ensure_ascii=False)[:500]}")
    print("variance", json.dumps(payload["variance"]))
    return payload


if __name__ == "__main__":
    main()
