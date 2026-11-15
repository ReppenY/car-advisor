"""Search-gate precision and recall from saved Part B trajectories. No new agent runs.

required and forbidden cases are the classification. optional cases are left
out of precision and recall, and still count toward the car-argument check.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_advisor.store import connect, resolve_cars  # noqa: E402

LABELS = ROOT / "eval" / "gate_labels.json"
RESULTS = ROOT / "eval" / "results.json"
OUT = ROOT / "eval" / "gate_metrics.json"


def families_of(conn, query: str) -> list[str]:
    status, payload = resolve_cars(conn, query)
    if status == "candidates":
        return list(payload)
    if status in ("rows", "mention"):
        return list(dict.fromkeys(row["family_id"] for row in payload))
    return []


def car_ok(conn, args: dict, label: dict) -> bool:
    car = str(args.get("car") or "").strip()
    allowed = label.get("car_any") or []
    if label.get("omit_car") and not label.get("car_family"):
        if not car:
            return True
        if not allowed:
            return False
        return bool(set(families_of(conn, car)) & set(allowed))
    wanted = label.get("car_family")
    if not wanted:
        return True
    if not car:
        return False
    return wanted in families_of(conn, car)


def score_run(conn, case_id: str, run: dict, label: dict) -> dict:
    searches = [call for call in run.get("tools") or [] if call["name"] == "search_passages"]
    searched = bool(searches)
    policy = label["search"]
    counted = policy in ("required", "forbidden")
    predicted = searched
    gold = policy == "required"
    car_checks = [car_ok(conn, call["args"], label) for call in searches]
    return {
        "case": case_id,
        "run": run["run"],
        "policy": policy,
        "searched": searched,
        "counted": counted,
        "gold": gold,
        "predicted": predicted,
        "false_allow": counted and predicted and not gold,
        "false_block": counted and gold and not predicted,
        "car_checked": len(car_checks),
        "car_ok": all(car_checks) if car_checks else None,
        "car_args": [call["args"].get("car") for call in searches],
    }


def rates(rows: list[dict]) -> dict:
    counted = [row for row in rows if row["counted"]]
    tp = sum(1 for row in counted if row["gold"] and row["predicted"])
    fp = sum(1 for row in counted if row["false_allow"])
    fn = sum(1 for row in counted if row["false_block"])
    checked = [row for row in rows if row["car_checked"]]
    car_hits = sum(1 for row in checked if row["car_ok"])
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_allows": fp,
        "false_blocks": fn,
        "car_arg_runs": len(checked),
        "car_arg_correct": car_hits,
        "car_arg_accuracy": round(car_hits / len(checked), 4) if checked else None,
    }


def main():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    conn = connect()
    rows = []
    try:
        for case in report["cases"]:
            label = labels[case["id"]]
            for run in case["runs"]:
                rows.append(score_run(conn, case["id"], run, label))
    finally:
        conn.close()
    payload = {"overall": rates(rows), "rows": rows}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overall = payload["overall"]
    print(
        f"gate precision={overall['precision']} recall={overall['recall']} "
        f"false_allows={overall['false_allows']} false_blocks={overall['false_blocks']} "
        f"car_arg={overall['car_arg_correct']}/{overall['car_arg_runs']}"
    )
    for row in rows:
        if row["false_allow"] or row["false_block"] or row["car_ok"] is False:
            print(
                f"  {row['case']} r{row['run']} policy={row['policy']} searched={row['searched']} "
                f"car={row['car_args']}"
            )
    return payload


if __name__ == "__main__":
    main()
