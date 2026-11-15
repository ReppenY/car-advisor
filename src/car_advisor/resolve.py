"""Name keys and edit distance.

Extraction does not use this. Ingestion uses it to flag look-alike families.
Query time uses it to resolve a name the agent passed.
"""

import re
from dataclasses import dataclass, field

from car_advisor.log import log

HEBREW_PREFIXES = set("והבלמשכ")
_NIQQUD = re.compile(r"[\u0591-\u05C7]")
_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_GERESH = str.maketrans({
    "'": None, "\u05f3": None, "`": None, "\u2019": None, "\u2018": None,
    '"': None, "\u05f4": None, "\u201c": None, "\u201d": None,
})


def strip_geresh(text: str) -> str:
    return (text or "").translate(_GERESH)


def compact(text: str) -> str:
    """Casefold, strip niqqud, drop geresh/gershayim, fold final letters."""
    text = _NIQQUD.sub("", text or "")
    text = strip_geresh(text)
    text = text.translate(_FINALS)
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def allowed_distance(length: int) -> int:
    """0 through 3 characters, 1 through 7, 2 from 8 up."""
    if length <= 3:
        return 0
    if length <= 7:
        return 1
    return 2


def damerau_levenshtein(left: str, right: str) -> int:
    """Adjacent transposition counts as one edit."""
    if left == right:
        return 0
    if not left or not right:
        return len(left) + len(right)
    prev2 = None
    prev = list(range(len(right) + 1))
    for i, ca in enumerate(left, start=1):
        curr = [i]
        for j, cb in enumerate(right, start=1):
            cost = 0 if ca == cb else 1
            best = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == right[j - 2] and left[i - 2] == cb:
                best = min(best, prev2[j - 2] + 1)
            curr.append(best)
        prev2, prev = prev, curr
    return prev[-1]


def title_blob(article: str) -> str:
    kept = []
    for line in article.splitlines():
        if line.startswith("source:"):
            kept.append(line)
        elif line.startswith("# ") and not any(item.startswith("# ") for item in kept):
            kept.append(line)
    return "\n".join(kept)


def name_in_title(name_he: str, name_en: str, article: str) -> bool:
    """A subject name has to show up in the title or the main heading, exactly."""
    hay = compact(title_blob(article))
    for name in (name_he, name_en):
        key = compact(name)
        if len(key) >= 2 and key in hay:
            return True
    return False


def english_parts(name_en: str) -> tuple[str, str]:
    def style(token: str) -> str:
        if any(ch.isdigit() for ch in token):
            return token.upper()
        return token[:1].upper() + token[1:].lower() if token else token

    tokens = (name_en or "").split()
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        one = style(tokens[0])
        return one, one
    return style(tokens[0]), " ".join(style(token) for token in tokens[1:])


@dataclass
class Match:
    status: str
    family_ids: list[str] = field(default_factory=list)
    near: list[tuple[str, str, int]] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    suggestions: list[tuple[str, str, str, int]] = field(default_factory=list)


def _alias(item) -> tuple[str, str, str]:
    if len(item) >= 3:
        return item[0], item[1], item[2] or ""
    return item[0], item[1], ""


def _kind_for(family_kinds: set[str]) -> str:
    if family_kinds == {"trim"}:
        return "trim"
    if family_kinds == {"make"}:
        return "make"
    return "family"


def _match_cap(query_len: int, alias: str, kind: str) -> int:
    """A make uses the longer string, so פג'ו can be one edit from פיג'ו."""
    span = max(query_len, len(alias)) if kind == "make" else query_len
    return allowed_distance(span)


def _suggestions(key: str, aliases: list) -> list[tuple[str, str, str, int]]:
    """Makes and models just past a match. Not matches."""
    cap = allowed_distance(len(key)) + 1
    best: dict[tuple[str, str], tuple[int, str, str]] = {}
    for alias, family, kind in (_alias(item) for item in aliases):
        if kind not in {"make", "model", "make_model"} or not alias:
            continue
        distance = damerau_levenshtein(key, alias)
        if not 0 < distance <= cap:
            continue
        group = "make" if kind == "make" else "car"
        current = best.get((family, group))
        if current is None or distance < current[0]:
            best[(family, group)] = (distance, alias, kind)
    rows = [
        (alias, family, kind, distance)
        for (family, _group), (distance, alias, kind) in best.items()
    ]
    rows.sort(key=lambda item: (item[3], item[1], item[0]))
    return rows[:8]


def match_aliases(query: str, aliases: list) -> Match:
    """Resolve one name. aliases are (compact key, family id) or (key, family, kind).

    Short names are exact only. Two families inside the allowed distance
    come back as candidates. A trim hit stays on that trim.
    A make alias uses the longer of the query and the alias for that distance.
    """
    key = compact(query)
    if not key:
        return Match("none")
    exact = [_alias(item) for item in aliases if _alias(item)[0] == key]
    if exact:
        families = list(dict.fromkeys(family for _alias_key, family, _kind in exact))
        kinds, keys = {}, {}
        for family in families:
            family_kinds = {kind for _alias_key, item_family, kind in exact if item_family == family}
            kinds[family] = _kind_for(family_kinds)
            keys[family] = key
        return Match("one" if len(families) == 1 else "candidates", families, kinds=kinds, keys=keys)

    query_limit = allowed_distance(len(key))
    best: dict[str, tuple[int, str, str]] = {}
    near: dict[str, tuple[int, str]] = {}
    for alias, family, kind in (_alias(item) for item in aliases):
        if not alias:
            continue
        distance = damerau_levenshtein(key, alias)
        cap = _match_cap(len(key), alias, kind)
        if cap and 0 < distance <= cap:
            current = best.get(family)
            better_kind = (
                current is not None
                and distance == current[0]
                and (
                    (current[2] == "trim" and kind != "trim")
                    or (current[2] == "make" and kind not in {"make", "trim"})
                )
            )
            if current is None or distance < current[0] or better_kind:
                best[family] = (distance, alias, kind)
        elif distance == query_limit + 1:
            current = near.get(family)
            if current is None or distance < current[0]:
                near[family] = (distance, alias)
    near_rows = [(alias, family, distance) for family, (distance, alias) in near.items() if family not in best]
    near_rows.sort(key=lambda item: item[2])
    kinds = {family: kind for family, (_distance, _alias_key, kind) in best.items()}
    keys = {family: alias for family, (_distance, alias, _kind) in best.items()}
    if not best:
        return Match("none", near=near_rows[:5], suggestions=_suggestions(key, aliases))
    if len(best) > 1:
        return Match("candidates", list(best), near_rows[:5], kinds, keys)
    return Match("one", list(best), near_rows[:5], kinds, keys)


def log_match(query: str, match: Match) -> None:
    for key, family, distance in match.near:
        log("near_miss", name=query, key=key, family=family, distance=distance)
    if match.status == "none" and compact(query):
        log("unresolved", name=query)


def possible_duplicates(labels: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """Family pairs whose names are within the allowed edit distance. Not merged."""
    found = []
    seen = set()
    for i, (left_id, left_key) in enumerate(labels):
        if not left_key:
            continue
        for right_id, right_key in labels[i + 1:]:
            if not right_key or left_id == right_id:
                continue
            pair = tuple(sorted((left_id, right_id)))
            if pair in seen:
                continue
            distance = damerau_levenshtein(left_key, right_key)
            limit = allowed_distance(max(len(left_key), len(right_key)))
            if distance <= limit:
                seen.add(pair)
                found.append((pair[0], pair[1], distance))
    return found


def hebrew_make(name_he: str) -> str:
    """The first word of a Hebrew name, when the name also has a model."""
    tokens = (name_he or "").split()
    if len(tokens) < 2:
        return ""
    return tokens[0]


def _text(car, name: str) -> str:
    try:
        return car[name] or ""
    except (KeyError, IndexError):
        return ""


def alias_records(cars: list) -> list[tuple[str, str, str]]:
    """Compact aliases from the catalog. The same key may point at two families."""
    seen = set()
    rows = []

    def add(text: str, family_id: str, kind: str):
        key = compact(text)
        if len(key) < 2 or not family_id:
            return
        item = (key, family_id, kind)
        if item in seen:
            return
        seen.add(item)
        rows.append(item)

    for car in cars:
        family_id = _text(car, "family_id")
        add(_text(car, "make"), family_id, "make")
        add(hebrew_make(_text(car, "name_he")), family_id, "make")
        add(f"{_text(car, 'make')} {_text(car, 'model')}", family_id, "make_model")
        add(_text(car, "model"), family_id, "model")
        add(_text(car, "name_he"), family_id, "article")
        add(_text(car, "name_en"), family_id, "article")
        add(_text(car, "trim_he"), family_id, "trim")
        add(_text(car, "trim_en"), family_id, "trim")
        model = _text(car, "model")
        hebrew = _text(car, "name_he").split()
        if len(model.split()) == 1 and len(hebrew) >= 2:
            add(hebrew[-1], family_id, "model")
    return rows
