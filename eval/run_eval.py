"""Run eval/cases.json on the live agent. Three runs per case.

Deterministic checks run first. An LLM judge scores only expected_behavior.
"""

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from car_advisor.eval_checks import car_problems, fact_problems, length_problems, session_is_short
from car_advisor.graph import graph, run_turn
from car_advisor.profile import mismatch_mentions
from car_advisor.llm import model
from car_advisor.log import log
from car_advisor.store import connect, resolve_cars

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "eval" / "cases.json"
CHECKS = ("trajectory", "tool_args", "cars", "facts", "length", "behavior")


class Verdict(BaseModel):
    ok: bool
    reason: str = Field(description="One sentence on the behavior only")


def families_of(conn, query: str) -> tuple[str, list[str]]:
    if not (query or "").strip():
        return "empty", []
    status, payload = resolve_cars(conn, query)
    if status == "candidates":
        return status, list(payload)
    if status in ("rows", "mention"):
        return status, list(dict.fromkeys(row["family_id"] for row in payload))
    return status, []


def call_query(call: dict) -> str:
    args = call["args"]
    if call["name"] == "find_cars":
        return " ".join(part for part in (args.get("make"), args.get("model")) if part)
    return str(args.get("car") or "")


def spec_holds(conn, call: dict, spec: dict) -> str | None:
    args = call["args"]
    if spec.get("omit_car") and str(args.get("car") or "").strip():
        return "passed car"
    if spec.get("files"):
        have = set(args.get("files") or [])
        missing = [name for name in spec["files"] if name not in have]
        if missing:
            return "missing " + ", ".join(missing)
    needs_resolve = any(key in spec for key in ("family", "family_any", "status", "candidates"))
    if not needs_resolve:
        return None
    status, families = families_of(conn, call_query(call))
    if "status" in spec and status != spec["status"]:
        return f"{call_query(call)!r} is {status or 'unresolved'}"
    if "candidates" in spec:
        if status != "candidates" or not set(spec["candidates"]) <= set(families):
            return f"{call_query(call)!r} is not ambiguous across both cars"
    wanted = spec.get("family_any") or ([spec["family"]] if spec.get("family") else [])
    if wanted and not (set(wanted) & set(families)):
        return f"{call_query(call)!r} resolved to {families or status}"
    return None


def arg_problems(conn, calls: list[dict], case: dict) -> list[str]:
    problems = []
    for spec in case.get("expected_tool_args") or []:
        matched = [call for call in calls if call["name"] == spec["tool"]]
        if spec.get("families_all"):
            if not matched:
                continue
            seen = set()
            for call in matched:
                seen.update(families_of(conn, call_query(call))[1])
            missing = [name for name in spec["families_all"] if name not in seen]
            if missing:
                problems.append(f"{spec['tool']} did not resolve {', '.join(missing)}")
            continue
        if not matched:
            continue
        results = [spec_holds(conn, call, spec) for call in matched]
        if spec.get("every"):
            problems.extend(f"{spec['tool']}: {item}" for item in results if item)
        elif not any(item is None for item in results):
            problems.append(f"{spec['tool']}: {next(item for item in results if item)}")
    group = case.get("tool_arg_any") or []
    if group and not any(_group_hit(conn, calls, spec) for spec in group):
        problems.append("none of the accepted tool arguments was used")
    return problems


def _group_hit(conn, calls, spec) -> bool:
    matched = [call for call in calls if call["name"] == spec["tool"]]
    return any(spec_holds(conn, call, spec) is None for call in matched)


def trajectory_problems(calls: list[dict], case: dict) -> list[str]:
    names = [call["name"] for call in calls]
    problems = []
    missing = [name for name in case.get("expected_tools_called") or [] if name not in names]
    if missing:
        problems.append("did not call " + ", ".join(missing))
    extra = [name for name in case.get("expected_tools_not_called") or [] if name in names]
    if extra:
        problems.append("called " + ", ".join(extra))
    any_of = case.get("expected_any") or []
    if any_of and not any(name in names for name in any_of):
        problems.append("expected one of " + ", ".join(any_of))
    return problems


def turn_slices(messages) -> list[list]:
    starts = [
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage) and isinstance(message.content, str)
    ]
    slices = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(messages)
        slices.append(messages[start:end])
    return slices


def tool_calls(messages) -> list[dict]:
    calls = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] in ("Plan", "FinalAnswer"):
                continue
            calls.append({"name": call["name"], "args": call.get("args") or {}})
    return calls


def tool_text(messages) -> str:
    return "\n".join(str(message.content) for message in messages if isinstance(message, ToolMessage))


def token_counts(messages) -> dict:
    incoming = outgoing = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        if not isinstance(usage, dict):
            usage = dict(usage)
        incoming += int(usage.get("input_tokens") or 0)
        outgoing += int(usage.get("output_tokens") or 0)
    return {"input": incoming, "output": outgoing}


def judge(transcript: str, behavior: str) -> dict:
    try:
        verdict = model().with_structured_output(Verdict).invoke(
            "Judge only the behavior instruction. Ignore tool names and writing style.\n"
            "ok is true only when the last assistant reply does what the behavior asks.\n\n"
            f"Behavior:\n{behavior}\n\nConversation:\n{transcript}"
        )
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        log("eval_behavior", ok=False, reason=reason)
        return {"pass": False, "reason": reason}
    log("eval_behavior", ok=verdict.ok, reason=verdict.reason)
    return {"pass": verdict.ok, "reason": verdict.reason}


def grade(conn, case: dict, calls: list[dict], answer: str, session_text: str, families: set[str], turns: list[str]) -> dict:
    question = turns[-1] if turns else ""
    checks = {
        "trajectory": _box(trajectory_problems(calls, case)),
        "tool_args": _box(arg_problems(conn, calls, case)),
        "cars": _box(car_problems(answer, families, case)),
        "facts": _box(fact_problems(answer, session_text, case)),
        "length": _box(length_problems(
            answer, question, category=case.get("category") or "", short=session_is_short(turns),
        )),
    }
    return checks


def _box(problems: list[str]) -> dict:
    return {"pass": not problems, "problems": problems}


def touched_families(conn, calls: list[dict]) -> set[str]:
    found = set()
    for call in calls:
        found.update(families_of(conn, call_query(call))[1])
    return found


def run_case(conn, case: dict, index: int) -> dict:
    session = f"eval-{case['id']}-r{index}-{uuid.uuid4().hex[:8]}"
    turns = case.get("turns") or [case["question"]]
    started = time.perf_counter()
    finals = []
    error = None
    try:
        for text in turns:
            finals.append(run_turn(session, text))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency = round(time.perf_counter() - started, 2)
    state = graph.get_state({"configurable": {"thread_id": session}}).values
    messages = state.get("messages") or []
    slices = turn_slices(messages)
    graded = slices[-1] if slices else []
    calls = tool_calls(graded)
    answer = finals[-1].answer if finals and finals[-1] else ""
    transcript = []
    for text, final in zip(turns, finals):
        transcript.append(f"User: {text}")
        transcript.append(f"Assistant: {final.answer if final else '(no answer)'}")
    families = touched_families(conn, calls)
    checks = grade(conn, case, calls, answer, tool_text(messages), families, turns)
    if error:
        for name in ("trajectory", "tool_args", "cars", "facts", "length"):
            checks[name]["pass"] = False
            checks[name]["problems"].append(error)
    joined = "\n".join(transcript)
    if not answer:
        checks["behavior"] = {"pass": False, "reason": error or "no answer"}
    else:
        checks["behavior"] = judge(joined, case["expected_behavior"])
    if case.get("max_mismatch_mentions") is not None:
        count = mismatch_mentions(joined)
        limit = case["max_mismatch_mentions"]
        if count > limit:
            checks["behavior"]["pass"] = False
            checks["behavior"]["reason"] = (
                f"{checks['behavior'].get('reason', '')} Mismatch reminders: {count}, limit {limit}."
            ).strip()
    passed = all(checks[name]["pass"] for name in CHECKS)
    return {
        "run": index,
        "session": session,
        "latency_s": latency,
        "tokens": token_counts(messages),
        "tools": calls,
        "answer": answer,
        "checks": checks,
        "pass": passed,
    }


def rates(runs: list[dict]) -> dict:
    total = len(runs) or 1
    summary = {}
    for name in CHECKS:
        got = sum(1 for run in runs if run["checks"][name]["pass"])
        summary[name] = f"{got}/{total}"
    got = sum(1 for run in runs if run["pass"])
    summary["all"] = f"{got}/{total}"
    return summary


def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.ids:
        cases = [case for case in cases if case["id"] in args.ids]
    conn = connect()
    results = ROOT / "eval" / f"results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report = {"runs_per_case": args.runs, "cases": []}
    try:
        for case in cases:
            runs = []
            for index in range(1, args.runs + 1):
                result = run_case(conn, case, index)
                runs.append(result)
                flags = " ".join(
                    f"{name}={'ok' if result['checks'][name]['pass'] else 'fail'}" for name in CHECKS
                )
                print(f"{case['id']} r{index} {flags}", flush=True)
            entry = {
                "id": case["id"],
                "category": case["category"],
                "pass_rate": rates(runs),
                "runs": runs,
            }
            report["cases"].append(entry)
            results.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if any("credit balance is too low" in json.dumps(run) for run in runs):
                print("stopped: Anthropic credit balance is too low", flush=True)
                break
    finally:
        conn.close()
    print(f"wrote {results}")
    for entry in report["cases"]:
        print(f"{entry['id']:24} {entry['pass_rate']['all']:6} {entry['pass_rate']}")


if __name__ == "__main__":
    main()
