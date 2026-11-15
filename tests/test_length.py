"""Length caps and the short-answer preference. No model calls."""

import json
from pathlib import Path

from car_advisor.eval_checks import classify_intent, length_problems, word_cap
from car_advisor.profile import asks_short, remember_style

CASES = json.loads((Path(__file__).parents[1] / "eval" / "cases.json").read_text(encoding="utf-8"))
BY_ID = {case["id"]: case for case in CASES}


def last_question(case_id: str) -> str:
    case = BY_ID[case_id]
    return (case.get("turns") or [case["question"]])[-1]


def test_short_is_remembered_for_the_session():
    assert asks_short("תענה בקצרה")
    assert asks_short("קצר")
    assert asks_short("keep it short")
    assert not asks_short("כמה עולה הפולו?")
    assert not asks_short("הקצרה")
    assert remember_style(None, "כמה עולה?") is None
    assert remember_style(None, "בקצרה, כמה עולה?") == "short"
    assert remember_style("short", "ומה עם היאריס?") == "short"


def test_intents_follow_the_question():
    assert classify_intent(last_question("polo-en"), "reviewed") == "direct"
    assert classify_intent(last_question("biggest-trunk"), "no-car") == "direct"
    assert classify_intent(last_question("suv-list-with-prices"), "no-car") == "list"
    assert classify_intent(last_question("suv-under-150"), "no-car") == "list"
    assert classify_intent(last_question("preference-broadens"), "follow-up") == "recommendation"
    assert classify_intent(last_question("tucson-sportage"), "compare") == "comparison"


def test_length_allows_one_question_and_caps_words():
    direct = "הפולו שנבחן עולה 129,000 שקלים."
    assert length_problems(direct, "כמה עולה הפולו?", category="reviewed") == []
    long = " ".join(["מילה"] * (word_cap("direct") + 1))
    assert length_problems(long, "כמה עולה הפולו?")
    two = "רוצה SUV? או עירוני?"
    problems = length_problems(two, "מה אתה ממליץ?")
    assert any("question marks" in item for item in problems)
    short_cap = word_cap("direct", short=True)
    almost = " ".join(["מילה"] * short_cap)
    assert length_problems(almost, "בקצרה כמה עולה?", short=True) == []
    assert length_problems(almost + " עוד", "בקצרה כמה עולה?", short=True)
