"""One question per review fact. The answer may use only what the tools returned."""

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from car_advisor.graph import graph, run_turn
from car_advisor.numbers import covers, find_mentions
from car_advisor.schemas import FinalAnswer
from car_advisor.test_tools import blocks, canonicals

load_dotenv(override=True)

RESULTS = Path(__file__).parents[2] / "results"


@dataclass
class Case:
    id: str
    question: str
    tools: list[str]
    files: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    sources_empty: bool = False
    sources_required: bool = False
    no_prices: bool = False
    require: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    named_car: bool = True


CASES = [
    Case(
        id="polo-price",
        question="כמה עולה פולקסווגן פולו בגרסת 110 כ\"ס שנבחנה?",
        tools=["find_cars", "search_passages"],
        files=["polo.md"],
        numbers=["129000"],
    ),
    Case(
        id="yaris-mazda-consumption",
        question="מה צריכת הדלק של טויוטה יאריס ושל מאזדה 2 במבחן ההשוואתי?",
        tools=["search_passages", "compare"],
        files=["yaris-mazda2.md"],
        numbers=["20.6", "14.5"],
    ),
    Case(
        id="trunk-compare",
        question="תשווה את נפח תא המטען של סקודה קודיאק ושל מיצובישי אאוטלנדר כשכל המושבים בשימוש.",
        tools=["compare"],
        files=["kodiaq.md", "outlander.md"],
        numbers=["289", "148"],
    ),
    Case(
        id="xpeng-price",
        question="כמה עולה אקספנג G6 בגרסת פרפורמנס שנבחנה?",
        tools=["find_cars", "search_passages"],
        files=["xpeng-g6.md"],
        numbers=["255000"],
    ),
    Case(
        id="unknown-car",
        question="מה המחיר של מאזדה CX-5?",
        tools=["find_cars"],
        require=["find_cars"],
        sources_empty=True,
        no_prices=True,
    ),
    Case(
        id="explorer-mention",
        question="במבחן של יונדאי פאליסייד, איזה מחיר רשום לפורד אקספלורר?",
        tools=["find_cars", "search_passages", "compare"],
        require=["find_cars"],
        files=["palisade.md"],
        numbers=["365000"],
    ),
    Case(
        id="model-y",
        question="מה המחיר של טסלה מודל Y?",
        tools=["find_cars"],
        require=["find_cars"],
        files=["xpeng-g6.md"],
        no_prices=True,
    ),
    Case(
        id="palisade-calligraphy",
        question="כמה עולה יונדאי פאליסייד בגרסת קליגרפי?",
        tools=["find_cars", "search_passages"],
        require=["find_cars"],
        files=["palisade.md"],
        numbers=["390000"],
    ),
    Case(
        id="mazda2-alias",
        question="מה צריכת הדלק של מאזדה2 במבחן?",
        tools=["search_passages", "compare"],
        files=["yaris-mazda2.md"],
        numbers=["14.5"],
    ),
    Case(
        id="no-car-trunk",
        question="מה נפח תא המטען במבחני הדרכים כשכל המושבים בשימוש?",
        tools=["search_passages", "compare"],
        sources_required=True,
        named_car=False,
    ),
]


def _calls_and_outputs(messages):
    calls, outputs = [], []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] not in ("FinalAnswer", "Plan"):
                calls.append(call)
        if message.__class__.__name__ == "ToolMessage" and not str(message.content).startswith("Answer"):
            if not str(message.content).startswith("Rejected") and not str(message.content).startswith("Plan"):
                if not str(message.content).startswith("Invalid"):
                    outputs.append(str(message.content))
    return calls, outputs


def _called(calls, names, files):
    matched = [call for call in calls if call["name"] in names]
    if not matched:
        return False
    with_files = [call for call in matched if call["args"].get("files")]
    if not files or not with_files:
        return True
    return any(all(file in call["args"]["files"] for file in files) for call in with_files)


def _cited_bodies(answer: FinalAnswer, tool_texts: list[str]) -> tuple[list[str], list[str]]:
    found = [(file, lines, body) for text in tool_texts for file, lines, body in blocks(text)]
    problems, bodies = [], []
    for source in answer.sources:
        match = next((body for file, lines, body in found if file == source.file and lines == source.lines), None)
        if match is None:
            problems.append(f"{source.file} lines {source.lines} was not returned by a tool")
        else:
            bodies.append(match)
    return problems, bodies


def _check(case: Case, answer: FinalAnswer | None, calls, outputs) -> list[str]:
    problems = []
    if answer is None:
        return ["no FinalAnswer"]
    names = [call["name"] for call in calls]
    if not _called(calls, case.tools, case.files):
        got = [(call["name"], call["args"]) for call in calls]
        problems.append(f"expected one of {case.tools} files={case.files}, got {got}")
    missing = [name for name in case.require if name not in names]
    if missing:
        problems.append(f"did not call {missing}")
    extra = [name for name in case.forbid if name in names]
    if extra:
        problems.append(f"called {extra}")
    searches = [call for call in calls if call["name"] == "search_passages"]
    if case.named_car:
        bare = [call for call in searches if not str(call["args"].get("car") or "").strip()]
        if searches and bare:
            problems.append("search_passages missing car")
    else:
        named = [call for call in searches if str(call["args"].get("car") or "").strip()]
        if named:
            problems.append("search_passages passed car on a question that names no car")
    if case.sources_empty and answer.sources:
        problems.append(f"sources should be empty, got {answer.sources}")
    if case.sources_required and not answer.sources:
        problems.append("expected a cited source")
    cited_files = {source.file for source in answer.sources}
    if case.files and not cited_files <= set(case.files):
        problems.append(f"cited {sorted(cited_files)}, allowed {case.files}")
    cite_problems, bodies = _cited_bodies(answer, outputs)
    problems.extend(cite_problems)
    allowed = set()
    for mention in find_mentions(case.question):
        allowed |= set(mention.forms)
    for body in bodies:
        for mention in find_mentions(body):
            allowed |= set(mention.forms)
    missing = covers(answer.answer, allowed)
    if missing:
        problems.append("numbers not in the cited tool text: " + ", ".join(missing))
    have = canonicals(answer.answer)
    for number in case.numbers:
        if number not in have:
            problems.append(f"answer missing {number}")
        if not any(number in canonicals(body) for body in bodies):
            problems.append(f"{number} is not in a cited tool passage")
    if case.no_prices:
        prices = [mention.surface for mention in find_mentions(answer.answer) if _price(mention.canonical)]
        if prices:
            problems.append("invented prices: " + ", ".join(prices))
    return problems


def _price(canonical: str) -> bool:
    try:
        return float(canonical) >= 10000
    except ValueError:
        return False


def _this_turn(messages):
    start = 0
    for index, message in enumerate(messages):
        if message.__class__.__name__ == "HumanMessage":
            start = index
    return messages[start:]


def run_case(case: Case) -> dict:
    session = f"case-{case.id}-{uuid.uuid4().hex[:8]}"
    final = run_turn(session, case.question)
    state = graph.get_state({"configurable": {"thread_id": session}}).values
    calls, outputs = _calls_and_outputs(_this_turn(state["messages"]))
    problems = _check(case, final, calls, outputs)
    return {
        "id": case.id,
        "question": case.question,
        "tools": [{"name": call["name"], "args": call["args"]} for call in calls],
        "answer": final.answer if final else None,
        "sources": [source.model_dump() for source in final.sources] if final else [],
        "problems": problems,
    }


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    chosen = sys.argv[1:]
    cases = [case for case in CASES if not chosen or case.id in chosen]
    RESULTS.mkdir(exist_ok=True)
    reports = [run_case(case) for case in cases]
    path = RESULTS / f"sessions-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [report for report in reports if report["problems"]]
    for report in reports:
        status = "FAIL" if report["problems"] else "ok"
        print(f"{status} {report['id']}")
        if report["problems"]:
            for problem in report["problems"]:
                print(f"   {problem}")
            print(f"   answer: {(report['answer'] or '')[:400]}")
    print(f"wrote {path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
