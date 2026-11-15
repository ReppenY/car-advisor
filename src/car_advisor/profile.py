"""Preference memory. A later explicit statement overrides an earlier vague one.

Hard preferences filter a recommendation. Soft ones rank it. Repeated questions
about cars outside a preference broaden it. A mismatch is mentioned once.
"""

import re

MISMATCH = (
    "לא רכב פנאי", "אינו רכב פנאי", "אינה רכב פנאי", "זה לא פנאי",
    "לא suv", "isn't an suv", "is not an suv", "not an suv",
    "לחזור", "go back",
)
_CHANGE = ("בעצם", "במקום", "actually", "instead")
_HARD = ("רק", "חייב", "only", "must", "בלי")
_YES = ("פתוח", "בטח", "כן", "yes", "sure", "open to")
_NO = ("לא פתוח", "רק רכב", "only suv", "just suv", "only an suv")
_SHORT = re.compile(r"(?<!\w)(בקצרה|קצר|short)(?!\w)", re.IGNORECASE)


def asks_short(text: str) -> bool:
    return bool(_SHORT.search(text or ""))


def remember_style(current: str | None, text: str) -> str | None:
    if current == "short" or asks_short(text):
        return "short"
    return current


def render_profile(profile: dict) -> str:
    lines = []
    for field, entry in (profile or {}).items():
        if not entry or not entry.get("value"):
            continue
        strength = entry.get("strength") or "soft"
        turn = entry.get("turn") or "?"
        lines.append(f"- {field}: {entry['value']} ({strength}, turn {turn}, \"{entry.get('quote', '')}\")")
    return "\n".join(lines) if lines else "(nothing yet)"


def overrides(current: dict | None, incoming: dict) -> bool:
    if not current:
        return True
    if incoming["turn"] < current.get("turn", 0):
        return False
    if incoming["strength"] == "hard":
        return True
    if current.get("strength") == "hard":
        return False
    if incoming["value"].casefold() == str(current.get("value", "")).casefold():
        return True
    return any(word in incoming.get("quote", "") for word in _CHANGE)


def apply_update(profile: dict, field: str, value: str, quote: str, strength: str, turn: int) -> None:
    if any(cue in (quote or "").casefold() for cue in _HARD):
        strength = "hard"
    elif strength == "hard":
        strength = "soft"
    incoming = {"value": value, "quote": quote, "strength": strength, "turn": turn}
    if overrides(profile.get(field), incoming):
        profile[field] = incoming


def preference_body(value: str) -> str | None:
    text = (value or "").casefold()
    if any(word in text for word in ("suv", "פנאי", "crossover")):
        return "suv"
    if any(word in text for word in ("city", "עירונ", "האצ", "hatch")):
        return "city"
    if "family" in text or "משפח" in text:
        return "family"
    return None


def body_group(segment: str | None) -> str | None:
    if segment in ("suv", "large-suv"):
        return "suv"
    if segment in ("city", "family"):
        return segment
    return None


def note_outside(profile: dict, segment: str | None, counts: dict) -> tuple[dict, dict]:
    """Count a direct question about a car outside the body preference. Broaden at two."""
    entry = profile.get("body")
    wanted = preference_body(entry["value"]) if entry else None
    got = body_group(segment)
    if not entry or not wanted or not got or wanted == got or "also considering" in entry["value"]:
        return profile, counts
    counts = dict(counts or {})
    key = f"body:{wanted}"
    counts[key] = counts.get(key, 0) + 1
    if counts[key] >= 2:
        profile = dict(profile)
        updated = dict(entry)
        updated["value"] = f"{entry['value']}; also considering {got}"
        profile["body"] = updated
    return profile, counts


def openness_answer(text: str) -> str | None:
    folded = (text or "").strip().casefold()
    if not folded or len(folded) > 80:
        return None
    if any(word in folded for word in _NO):
        return "no"
    if any(word in folded for word in _YES):
        return "yes"
    return None


def record_openness(profile: dict, text: str, asked: bool, answered: bool) -> tuple[dict, bool]:
    if not asked or answered:
        return profile, answered
    answer = openness_answer(text)
    entry = profile.get("body")
    if answer is None or not entry:
        return profile, answered
    profile = dict(profile)
    updated = dict(entry)
    if answer == "yes" and "also considering" not in updated["value"]:
        updated["value"] = f"{updated['value']}; also considering other types"
    if answer == "no":
        updated["strength"] = "hard"
    profile["body"] = updated
    return profile, True


def preference_context(kind: str, profile: dict, mentioned: dict, asked: bool, counts: dict) -> tuple[str, list[str], bool]:
    if kind != "recommendation":
        return (
            "Direct question about specific cars. Answer only what was asked. "
            "Do not add the segment or the fuel. Do not say the car is a city car, a hatchback, or not an SUV. "
            "Do not mention the saved preference and do not ask the user to go back.",
            [],
            False,
        )
    lines = ["Recommendation. Use the preference profile.", render_profile(profile)]
    pending = []
    ask = False
    body = (profile or {}).get("body") or {}
    value = body.get("value") or ""
    outside = any(count > 0 for key, count in (counts or {}).items() if key.startswith("body:"))
    if body and "also considering" in value:
        lines.append(
            "The preference was broadened. Recommend from every type it names, city cars and SUVs. "
            "Do not filter to the original type and do not ask whether they are open to other types."
        )
    elif outside and not mentioned.get("body"):
        lines.append(
            f"You may mention once that a suggestion is outside {value}. Do not mention that mismatch again."
        )
        pending.append("body")
        if not asked:
            lines.append("Ask once whether they are open to other body types. Do not ask this again.")
            ask = True
    elif outside and mentioned.get("body"):
        lines.append(f"Do not mention a mismatch with {value}. It was already said.")
    elif body and body.get("strength") == "hard":
        lines.append("This body preference is a hard constraint. Pass it as the find_cars segment filter.")
    elif body:
        lines.append("This body preference is soft. Rank matching cars first and keep the others. Do not filter them out.")
    return "\n".join(lines), pending, ask


def hard_filters(profile: dict) -> dict:
    """Arguments a hard constraint adds to find_cars. Soft preferences add none."""
    from car_advisor.numbers import find_mentions

    args = {}
    body = (profile or {}).get("body") or {}
    if body.get("strength") == "hard" and "also considering" not in (body.get("value") or ""):
        kind = preference_body(body.get("value") or "")
        if kind:
            args["segment"] = kind
    budget = (profile or {}).get("budget") or {}
    if budget.get("strength") == "hard":
        mentions = find_mentions(str(budget.get("value") or ""))
        if mentions and mentions[0].canonical.isdigit():
            args["price_max"] = int(mentions[0].canonical)
    return args


def mismatch_mentions(transcript: str) -> int:
    count = 0
    for line in (transcript or "").splitlines():
        if not line.startswith("Assistant:"):
            continue
        folded = line.casefold()
        if any(phrase in folded for phrase in MISMATCH):
            count += 1
    return count
