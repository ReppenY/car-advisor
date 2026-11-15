"""Answer checks on saved Part B runs. Replays search and compare. No new agent runs.

Judge scores are written only with --judge. They stay provisional until the
calibration table has been labeled by hand.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_advisor.store import connect  # noqa: E402
from car_advisor.tools import TOOLS_BY_NAME  # noqa: E402

RESULTS = ROOT / "eval" / "results.json"
CASES = ROOT / "eval" / "cases.json"
GATES = ROOT / "eval" / "gate_labels.json"
OUT = ROOT / "eval" / "answer_metrics.json"
CALIBRATION = ROOT / "eval" / "calibration.json"

SOURCE = re.compile(r"\[source:\s*([\w.-]+\.md)\s+lines\s+(\d+)\s*-\s*(\d+)\]", re.I)
UNIT = re.compile(
    r"""
    (?:(?P<shekel>₪)\s*)?
    (?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)
    (?:\s*(?P<thou>אלף))?
    (?:\s*(?P<unit>
        שקלים|שקל|ש"ח|nis|shekels?|
        כ"ס|hp|
        ליטרים|ליטר|ל'|liters?|
        ק"מ/ל'|ק"מ/ליטר|ק"מ\s*לליטר|km/l|
        נוסעים|מושבים|seats
    ))?
    """,
    re.VERBOSE | re.IGNORECASE,
)
TESTED = re.compile(r"נבחנ|tested", re.I)
LISTED = re.compile(r"רשום|מוזכר|listed|במבחן של|בסקירה", re.I)
NEGATION = re.compile(
    r"לא |אין |אינו |אינה |אינם |אינן |מעל |יקר|הזול|not an|isn't|aren't|does not|doesn't|above|over |hatchback|האצ",
    re.I,
)
LISTED_PRICES = {
    ("fordexplorer", "365000"),
    ("fordexplorer", "390000"),
    ("chevrolettraverse", "340000"),
    ("chevrolettraverse", "380000"),
}


def normalize(text: str) -> str:
    return (text or "").replace("״", '"').replace("׳", "'").replace("”", '"').replace("’", "'")


def unit_numbers(text: str) -> list[dict]:
    text = normalize(text)
    found = []
    for match in UNIT.finditer(text):
        if not (match.group("shekel") or match.group("thou") or match.group("unit")):
            continue
        raw = match.group("num").replace(",", "")
        value = float(raw)
        if match.group("thou"):
            value *= 1000
        canonical = str(int(value)) if value == int(value) else str(value)
        unit = match.group("unit") or ""
        if match.group("shekel"):
            unit = unit or "₪"
        elif match.group("thou"):
            unit = unit or "אלף"
        found.append({
            "surface": match.group(0).strip(),
            "canonical": canonical,
            "start": match.start(),
            "unit": unit,
        })
    return found


def catalog():
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM cars").fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def names_for(rows: list[dict]) -> list[tuple[str, str, str | None]]:
    found = []
    for row in rows:
        for field in ("name_he", "name_en", "trim_he", "trim_en"):
            text = (row[field] or "").strip()
            if len(text) >= 2:
                found.append((text, row["family_id"], text if field.startswith("trim") else None))
        full = f"{row['make']} {row['model']}".strip()
        if len(full) >= 3:
            found.append((full, row["family_id"], None))
        for text in (row["name_he"], row["name_en"], row["model"]):
            token = (text or "").split()[-1] if text else ""
            if len(token) >= 2 and not token.isdigit():
                found.append((token, row["family_id"], None))
    found.sort(key=lambda item: len(item[0]), reverse=True)
    return found


def closest(window: str, name: str, number_at: int):
    best = None
    cursor = 0
    while True:
        at = window.find(name, cursor)
        if at < 0:
            return best
        distance = abs(at - number_at)
        if best is None or distance < best:
            best = distance
        cursor = at + max(len(name), 1)
    return best


def attribute(answer: str, start: int, names: list[tuple[str, str, str | None]]):
    origin = max(0, start - 180)
    window = answer[origin: start + 80]
    number_at = start - origin
    best = None
    for name, family, trim in names:
        distance = closest(window, name, number_at)
        if distance is None:
            continue
        if best is None or distance < best[0] or (distance == best[0] and len(name) > len(best[3])):
            best = (distance, family, trim, name)
    if best is None:
        return None
    return {"family": best[1], "trim": best[2], "name": best[3], "window": window}


def family_rows(rows, family, trim):
    own = [row for row in rows if row["family_id"] == family]
    if trim:
        trimmed = [
            row for row in own
            if trim == (row["trim_he"] or "") or trim == (row["trim_en"] or "")
        ]
        if trimmed:
            return trimmed
    return own


def value_matches(row, canonical: str) -> bool:
    if row["price_nis"] is not None and str(row["price_nis"]) == canonical:
        return True
    if row["seats"] is not None and str(row["seats"]) == canonical:
        return True
    return False


def replay(calls: list[dict]) -> list[dict]:
    passages = []
    for call in calls:
        if call["name"] not in ("search_passages", "compare"):
            continue
        body = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        passages.extend(parse_passages(str(body)))
    return passages


def parse_passages(body: str) -> list[dict]:
    marks = list(SOURCE.finditer(body))
    parsed = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
        parsed.append({
            "file": mark.group(1),
            "start": int(mark.group(2)),
            "end": int(mark.group(3)),
            "text": body[mark.end():end],
        })
    return parsed


def number_forms(mention: dict) -> list[str]:
    forms = [mention["surface"]]
    canonical = mention["canonical"]
    if len(canonical) >= 2:
        forms.append(canonical)
        if canonical.isdigit():
            forms.append(f"{int(canonical):,}")
            if int(canonical) % 1000 == 0:
                forms.append(f"{int(canonical) // 1000} אלף")
    return forms


def near_owner(text: str, mention: dict, owner: str) -> bool:
    if not owner or owner not in text:
        return False
    for form in number_forms(mention):
        cursor = 0
        while True:
            at = text.find(form, cursor)
            if at < 0:
                break
            window = text[max(0, at - 70): at + len(form) + 70]
            if owner in window:
                return True
            cursor = at + max(len(form), 1)
    return False


def number_in_text(canonical: str, text: str) -> bool:
    mention = {"surface": canonical, "canonical": canonical}
    return any(form in normalize(text) or form in normalize(text).replace(",", "") for form in number_forms(mention))


def listed_as_tested(family: str, canonical: str, window: str) -> bool:
    if (family, canonical) not in LISTED_PRICES:
        return False
    return bool(TESTED.search(window)) and not LISTED.search(window)


def check_numbers(answer: str, rows: list[dict], names, passages: list[dict]) -> list[dict]:
    problems = []
    for mention in unit_numbers(answer):
        owner = attribute(answer, mention["start"], names)
        if owner is None:
            problems.append({**mention, "problem": "unattributed"})
            continue
        own = family_rows(rows, owner["family"], owner["trim"])
        matched = [row for row in own if value_matches(row, mention["canonical"])]
        files = {row["file"] for row in own}
        own_names = [name for name, family, _trim in names if family == owner["family"]]
        in_own_passage = any(
            passage["file"] in files and any(near_owner(passage["text"], mention, name) for name in own_names)
            for passage in passages
        )
        if listed_as_tested(owner["family"], mention["canonical"], owner["window"]):
            problems.append({**mention, "family": owner["family"], "problem": "listed price presented as tested"})
            continue
        if matched or in_own_passage:
            continue
        other_family = any(
            row["family_id"] != owner["family"] and value_matches(row, mention["canonical"])
            for row in rows
        )
        elsewhere = [
            passage["file"] for passage in passages
            if number_in_text(mention["canonical"], passage["text"])
        ]
        kind = "misattributed" if other_family or elsewhere else "unmatched"
        problems.append({
            **mention,
            "family": owner["family"],
            "problem": kind,
            "seen_in": elsewhere,
        })
    return problems


def check_citations(answer: str, passages: list[dict]) -> list[dict]:
    bad = []
    for mark in SOURCE.finditer(answer or ""):
        file, start, end = mark.group(1), int(mark.group(2)), int(mark.group(3))
        overlap = [
            passage for passage in passages
            if passage["file"] == file and passage["start"] <= end and start <= passage["end"]
        ]
        sentence_start = max(0, answer.rfind("\n", 0, mark.start()) + 1)
        sentence = answer[sentence_start:mark.end()]
        numbers = unit_numbers(sentence)
        contains = True
        if numbers:
            contains = all(number_in_text(item["canonical"], " ".join(item["text"] for item in overlap)) for item in numbers)
        if not overlap:
            bad.append({"file": file, "lines": f"{start}-{end}", "problem": "not in this run's passages"})
        elif not contains:
            bad.append({"file": file, "lines": f"{start}-{end}", "problem": "cited passage does not contain the number"})
    return bad


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text or "")
    return [part.strip() for part in parts if len(part.strip()) >= 12]


def recommended_families(answer: str, names) -> set[str]:
    chosen = set()
    for sentence in sentences(answer):
        if NEGATION.search(sentence):
            continue
        for name, family, _trim in names:
            if len(name) >= 3 and name in sentence:
                chosen.add(family)
    return chosen


def list_scores(answer: str, rows: list[dict], names, spec: dict) -> dict:
    qualifying = set()
    for row in rows:
        if row["role"] != "subject":
            continue
        if spec.get("segments") and row["segment"] not in spec["segments"]:
            continue
        if spec.get("budget") is not None and (row["price_nis"] is None or row["price_nis"] > spec["budget"]):
            continue
        qualifying.add(row["family_id"])
    recommended = recommended_families(answer, names)
    both = recommended & qualifying
    if recommended:
        precision = len(both) / len(recommended)
    else:
        precision = 1.0 if not qualifying else None
    recall = len(both) / len(qualifying) if qualifying else 1.0
    violations = sorted(recommended - qualifying)
    return {
        "precision": None if precision is None else round(precision, 4),
        "recall": round(recall, 4),
        "recommended": sorted(recommended),
        "qualifying": sorted(qualifying),
        "constraint_violations": violations,
    }


def score_runs() -> dict:
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in json.loads(CASES.read_text(encoding="utf-8"))}
    gates = json.loads(GATES.read_text(encoding="utf-8"))
    rows = catalog()
    names = names_for(rows)
    scored = []
    for case in report["cases"]:
        spec = gates.get(case["id"], {})
        for run in case["runs"]:
            passages = replay(run.get("tools") or [])
            answer = run.get("answer") or ""
            numbers = check_numbers(answer, rows, names, passages)
            citations = check_citations(answer, passages)
            lists = list_scores(answer, rows, names, spec["list"]) if spec.get("list") else None
            must = cases[case["id"]].get("must_include") or []
            present = [
                item["number"] for item in must
                if item["number"] in {mention["canonical"] for mention in unit_numbers(answer)}
                or item["number"] in re.sub(r",", "", answer)
            ]
            scored.append({
                "case": case["id"],
                "category": case["category"],
                "run": run["run"],
                "number_problems": numbers,
                "bad_citations": citations,
                "car_list": lists,
                "must_include_present": present,
                "must_include_total": len(must),
            })
    return {"runs": scored}


def claims_of(answer: str) -> list[str]:
    return sentences(answer)[:8]


def judge_one(answer: str, question: str, passages: list[dict], must: list[dict]) -> dict:
    from pydantic import BaseModel, Field

    from car_advisor.llm import model

    class Claim(BaseModel):
        index: int
        label: str = Field(description="supported, not_supported, or contradicted")

    class Part(BaseModel):
        part: str
        answered: bool

    class Review(BaseModel):
        claims: list[Claim]
        parts: list[Part]

    claim_list = claims_of(answer)
    passage_text = "\n\n".join(
        f"[source: {item['file']} lines {item['start']}-{item['end']}]\n{item['text']}" for item in passages
    ) or "(no passages were retrieved)"
    numbered = "\n".join(f"{index}. {claim}" for index, claim in enumerate(claim_list))
    prompt = (
        "You see only the passages below. Do not use outside knowledge.\n"
        "Label each claim supported, not_supported, or contradicted.\n"
        "supported means the passages state it. contradicted means a passage says the opposite.\n"
        "Then split the question into the parts it asks and mark each answered or not.\n\n"
        f"Question:\n{question}\n\nPassages:\n{passage_text}\n\nClaims:\n{numbered}"
    )
    review = model().with_structured_output(Review).invoke(prompt)
    labels = {item.index: item.label for item in review.claims}
    judged = [
        {"claim": claim, "label": labels.get(index, "not_supported")}
        for index, claim in enumerate(claim_list)
    ]
    answered = sum(1 for part in review.parts if part.answered)
    present = sum(1 for item in must if item["number"] in re.sub(r",", "", answer))
    total = len(review.parts) + len(must)
    fact_recall = (answered + present) / total if total else None
    supported = sum(1 for item in judged if item["label"] == "supported")
    return {
        "claims": judged,
        "parts": [{"part": part.part, "answered": part.answered} for part in review.parts],
        "claim_precision": round(supported / len(judged), 4) if judged else None,
        "fact_recall": round(fact_recall, 4) if fact_recall is not None else None,
        "provisional": True,
    }


def attach_judge(payload: dict, limit: int | None) -> dict:
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in json.loads(CASES.read_text(encoding="utf-8"))}
    by_key = {(item["case"], item["run"]): item for item in payload["runs"]}
    sample = []
    judged = 0
    for case in report["cases"]:
        for run in case["runs"]:
            if limit is not None and judged >= limit:
                break
            if not run.get("answer"):
                continue
            passages = replay(run.get("tools") or [])
            review = judge_one(
                run["answer"],
                case.get("question") or " ".join(case.get("turns") or []),
                passages,
                cases[case["id"]].get("must_include") or [],
            )
            by_key[(case["id"], run["run"])]["judge"] = review
            if len(sample) < 20:
                for item in review["claims"]:
                    sample.append({
                        "case": case["id"],
                        "run": run["run"],
                        "claim": item["claim"],
                        "judge": item["label"],
                        "yours": "",
                    })
            judged += 1
            print(f"judged {case['id']} r{run['run']} claims={review['claim_precision']} facts={review['fact_recall']}", flush=True)
        if limit is not None and judged >= limit:
            break
    payload["judge_provisional"] = True
    payload["calibration"] = sample
    CALIBRATION.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-limit", type=int, default=0)
    args = parser.parse_args()
    payload = score_runs()
    if args.judge:
        payload = attach_judge(payload, args.judge_limit or None)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    problems = sum(len(item["number_problems"]) for item in payload["runs"])
    citations = sum(len(item["bad_citations"]) for item in payload["runs"])
    print(f"number problems={problems} bad citations={citations}")
    for item in payload["runs"]:
        if item["number_problems"] or item["bad_citations"] or (item["car_list"] and item["car_list"]["constraint_violations"]):
            print(f"  {item['case']} r{item['run']} numbers={item['number_problems']} citations={item['bad_citations']} list={item['car_list']}")
    return payload


if __name__ == "__main__":
    main()
