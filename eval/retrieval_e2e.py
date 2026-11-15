"""Agent eval with hybrid, dense-only, and BM25-only search.

The default search() is left as hybrid. This process points the tool module
at search_with for one method at a time, then puts the original back.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv  # noqa: E402

import car_advisor.tools as tools  # noqa: E402
from car_advisor.graph import graph  # noqa: E402
from car_advisor.store import connect  # noqa: E402
from retrieval_ablation import search_with  # noqa: E402
from run_eval import CHECKS, run_case  # noqa: E402

CASES = ROOT / "eval" / "cases.json"
OUT = ROOT / "eval" / "retrieval_e2e.json"
METHODS = ("hybrid", "dense", "bm25")
ORIGINAL = tools.search


def install(method: str) -> None:
    def wrapped(query, *, k=4, files=None, root=None):
        return search_with(query, k=k, files=files, root=root, method=method)

    tools.search = wrapped


def searches(session: str) -> int:
    messages = graph.get_state({"configurable": {"thread_id": session}}).values.get("messages") or []
    count = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] in ("search_passages", "compare"):
                count += 1
    return count


def summarize(method: str, runs: list[dict]) -> dict:
    total = len(runs) or 1
    passed = {name: sum(1 for run in runs if run["checks"][name]["pass"]) for name in CHECKS}
    tokens_in = sum(run["tokens"]["input"] for run in runs)
    tokens_out = sum(run["tokens"]["output"] for run in runs)
    search_calls = sum(searches(run) for run in runs)
    questions = sum(run["questions"] for run in runs) or 1
    return {
        "method": method,
        "cases": len(runs),
        "rates": {name: f"{passed[name]}/{total}" for name in CHECKS},
        "all": f"{sum(1 for run in runs if run['pass'])}/{total}",
        "searches_per_case": round(search_calls / total, 2),
        "searches_per_question": round(search_calls / questions, 2),
        "tokens_per_case": round((tokens_in + tokens_out) / total),
        "input_tokens_per_case": round(tokens_in / total),
        "output_tokens_per_case": round(tokens_out / total),
    }


def main():
    load_dotenv(override=True)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    conn = connect()
    report = {"runs_per_case": 1, "note": "One run. Default search() was not changed.", "methods": []}
    try:
        for method in METHODS:
            install(method)
            runs = []
            print(f"method {method}", flush=True)
            for case in cases:
                result = run_case(conn, case, 1)
                result["id"] = case["id"]
                result["questions"] = len(case.get("turns") or [case["question"]])
                result["searches"] = searches(result["session"])
                runs.append(result)
                flags = " ".join(
                    f"{name}={'ok' if result['checks'][name]['pass'] else 'fail'}" for name in CHECKS
                )
                print(f"{method} {case['id']} {flags} searches={result['searches']}", flush=True)
                if "credit balance is too low" in json.dumps(result):
                    print("stopped: Anthropic credit balance is too low", flush=True)
                    report["methods"].append({"method": method, "summary": summarize(method, runs), "runs": runs})
                    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    return
            entry = {"method": method, "summary": summarize(method, runs), "runs": runs}
            report["methods"].append(entry)
            OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(method, entry["summary"], flush=True)
    finally:
        tools.search = ORIGINAL
        conn.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
