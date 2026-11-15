import re
from dataclasses import dataclass
from decimal import Decimal

# Longest shape first. אלף / k multiply by 1000. שקלים is the plural these articles use.
MENTION = re.compile(
    r"""
    (?P<cur>₪\s*)?
    (?P<num>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
      | \d+(?:\.\d+)?
    )
    (?P<thou>\s*(?:אלף|[kK]))?
    (?P<unit>\s*(?:שקלים|שקל|ש״ח|ש"ח|שח))?
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Mention:
    surface: str
    canonical: str
    forms: frozenset[str]


def _canonical(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _thousand_spellings(value: Decimal) -> set[str]:
    scaled = value / Decimal(1000)
    tenths = scaled.quantize(Decimal("0.1"))
    if tenths * 1000 != value:
        return set()
    token = _canonical(tenths)
    return {f"{token} אלף", f"{token}אלף", f"{token}k", f"{token}K"}


def forms_for(value: Decimal, surface: str, *, money: bool) -> frozenset[str]:
    plain = _canonical(value)
    forms = {plain, surface.strip()}
    if value == value.to_integral_value():
        whole = int(value)
        forms.add(str(whole))
        forms.add(f"{whole:,}")
    if value >= 1000:
        forms |= _thousand_spellings(value)
    if money:
        grouped = f"{int(value):,}" if value == value.to_integral_value() else plain
        forms.add(f"₪{plain}")
        forms.add(f"₪{grouped}")
        forms.add(f"{grouped} שקל")
        forms.add(f"{grouped} שקלים")
        forms.add(f'{grouped} ש"ח')
    return frozenset(form for form in forms if form)


def parse_mention(match: re.Match) -> Mention:
    raw_num = match.group("num").replace(",", "")
    value = Decimal(raw_num)
    multiplied = bool(match.group("thou"))
    if multiplied:
        value *= 1000
    money = multiplied or bool(match.group("cur") or match.group("unit"))
    surface = match.group(0).strip()
    return Mention(surface, _canonical(value), forms_for(value, surface, money=money))


def _skip(text: str, match: re.Match) -> bool:
    """Model tokens (G6, EX5, CX-5) and list markers (1. 2.) are not claims.

    A hyphen in ב-411,000 or ו-411 אלף is a Hebrew prefix, so a grouped
    price or a price with a unit is kept. ל-6 is a seat count. CX-5 is a model.
    """
    start = match.start()
    end = match.end()
    if start and text[start - 1].isalnum():
        return True
    if end < len(text) and text[end].isalnum():
        return True
    num = match.group("num")
    priced = bool(match.group("thou") or match.group("cur") or match.group("unit"))
    if priced or "," in num or "." in num or len(num) > 2:
        return False
    if start and text[start - 1] in "-.":
        before = text[start - 2] if start >= 2 else ""
        if not before or (before.isascii() and before.isalnum()):
            return True
    return bool(re.match(r"[\.\)]\s", text[end:]))


def find_mentions(text: str) -> list[Mention]:
    text = text or ""
    return [parse_mention(match) for match in MENTION.finditer(text) if not _skip(text, match)]


def covers(answer: str, allowed: set[str]) -> list[str]:
    """Number mentions in the answer whose forms never appear in the allowed set."""
    missing = []
    for mention in find_mentions(answer):
        if mention.forms.isdisjoint(allowed):
            missing.append(mention.surface)
    return missing
