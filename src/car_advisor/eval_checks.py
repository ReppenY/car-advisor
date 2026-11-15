"""Deterministic checks for the agent eval. No model calls."""

import re

from car_advisor.numbers import find_mentions
from car_advisor.profile import asks_short

# Words, then the same cap halved when the session asked for a short answer.
WORD_CAPS = {
    "direct": 60,
    "list": 90,
    "recommendation": 120,
    "comparison": 160,
}
_PROS_EXTRA = 40
_RECOMMEND = ("ממליץ", "המלצ", "תמליץ", "recommend")
_LIST = ("תן לי", "רשימ", "איזה ", "אילו ")
_COMPARE = ("השוו", "השוואה", " לעומת ", " מול ", "compare", "versus")
_PROS = re.compile(r"(?<!\w)(יתרונות|חסרונות|pros|cons)(?!\w)", re.IGNORECASE)


def mention_windows(answer: str, number: str, radius: int = 180) -> list[str]:
    windows = []
    for mention in find_mentions(answer):
        if mention.canonical != number:
            continue
        start = answer.find(mention.surface)
        if start < 0:
            continue
        windows.append(answer[max(0, start - radius): start + len(mention.surface) + radius])
    return windows


def attribute_problems(answer: str, rules: list[dict]) -> list[str]:
    """A number is tied to its owner. A required number has to appear.

    not_owner fails only when the owner is absent, so an answer may mention
    both the Explorer and Calligraphy as long as the Explorer is named.
    """
    problems = []
    for rule in rules:
        windows = mention_windows(answer or "", rule["number"])
        if not windows:
            if rule.get("required", True):
                problems.append(f"{rule['number']} missing")
            continue
        owners = rule.get("owner") or []
        banned = rule.get("not_owner") or []
        for window in windows:
            owned = any(owner in window for owner in owners)
            if owners and not owned:
                hit = next((name for name in banned if name in window), None)
                if hit:
                    problems.append(f"{rule['number']} is attributed to {hit}")
                else:
                    problems.append(f"{rule['number']} is not attributed to {owners[0]}")
    return problems


def numbers_in(answer: str) -> set[str]:
    return {mention.canonical for mention in find_mentions(answer or "")}


def classify_intent(question: str, category: str = "") -> str:
    text = question or ""
    if category == "compare" or any(cue in text for cue in _COMPARE):
        return "comparison"
    if any(cue in text for cue in _RECOMMEND):
        return "recommendation"
    if any(cue in text for cue in _LIST):
        return "list"
    return "direct"


def asks_pros(question: str) -> bool:
    return bool(_PROS.search(question or ""))


def word_cap(intent: str, *, short: bool = False, pros: bool = False) -> int:
    cap = WORD_CAPS[intent]
    if short:
        cap //= 2
    if pros:
        cap += _PROS_EXTRA
    return cap


def length_problems(answer: str, question: str, *, category: str = "", short: bool = False) -> list[str]:
    intent = classify_intent(question, category)
    cap = word_cap(intent, short=short, pros=asks_pros(question))
    words = len((answer or "").split())
    problems = []
    if words > cap:
        problems.append(f"{words} words, {intent} allows {cap}")
    marks = (answer or "").count("?") + (answer or "").count("؟")
    if marks > 1:
        problems.append(f"{marks} question marks")
    return problems


def session_is_short(turns: list[str]) -> bool:
    return any(asks_short(text) for text in turns)


def fact_problems(answer: str, tool_text: str, case: dict) -> list[str]:
    have = numbers_in(answer)
    problems = []
    for item in case.get("must_include") or []:
        if item["number"] not in have:
            problems.append(f"answer missing {item['number']}")
        elif item.get("file") and item["file"] not in tool_text:
            problems.append(f"{item['number']} was not returned from {item['file']}")
    groups = case.get("must_include_any") or []
    if groups and not any(_group_ok(have, tool_text, group) for group in groups):
        problems.append("none of the fact groups matched")
    for number in case.get("must_not_include") or []:
        if number in have:
            problems.append(f"answer includes {number}")
    if case.get("no_prices"):
        prices = [number for number in have if _price(number)]
        if prices:
            problems.append("invented prices: " + ", ".join(sorted(prices)))
    problems.extend(attribute_problems(answer, case.get("attribute") or []))
    return problems


def _group_ok(have: set[str], tool_text: str, group: list[dict]) -> bool:
    for item in group:
        if item["number"] not in have:
            return False
        if item.get("file") and item["file"] not in tool_text:
            return False
    return True


def _price(canonical: str) -> bool:
    try:
        return float(canonical) >= 10000
    except ValueError:
        return False


def car_problems(answer: str, families: set[str], case: dict) -> list[str]:
    expected = case.get("expected_cars") or {}
    if not expected:
        return []
    names = expected.get("any_name") or []
    family = expected.get("family")
    named = any(name in (answer or "") for name in names)
    found = family in families if family else False
    if names and family:
        if not named and not found:
            return [f"answer does not mention {names[0]}"]
        return []
    if names and not named:
        return [f"answer does not mention {names[0]}"]
    if family and not found:
        return [f"tools did not resolve {family}"]
    return []
