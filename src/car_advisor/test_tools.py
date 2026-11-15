"""What each tool must return, taken from the saved reviews. No model calls."""

import re

from car_advisor.numbers import find_mentions
from car_advisor.paths import ARTICLES
from car_advisor.tools import compare, find_cars, search_passages

SOURCE = re.compile(r"\[source: (\S+) lines (\d+-\d+)\]")

# Cars named in a review only as rivals. They were not the car tested.
NOT_REVIEWED = [
    ("Tesla", "Model Y"),
    ("Toyota", "Highlander"),
    ("Kia", "Telluride"),
    ("Subaru", "Ascent"),
    ("Hyundai", "Santa Fe"),
    ("Ford", "Explorer"),
    ("Chevrolet", "Traverse"),
    ("Xpeng", "G9"),
    ("Xpeng", "P7"),
    ("Mazda", "CX-5"),
]


def blocks(text: str) -> list[tuple[str, str, str]]:
    found = []
    for part in re.split(r"(?=\[source: )", text):
        match = SOURCE.search(part)
        if match:
            found.append((match.group(1), match.group(2), part[match.end():]))
    return found


def canonicals(text: str) -> set[str]:
    return {mention.canonical for mention in find_mentions(text)}


def _prices(text: str) -> set[str]:
    return {item for item in canonicals(text) if item.isdigit() and int(item) >= 10000}


def test_search_cites_the_passage_that_holds_the_fact():
    polo = search_passages.invoke({"query": "מחיר גרסת המבחן", "files": ["polo.md"]})
    polo_blocks = [body for file, _, body in blocks(polo) if file == "polo.md"]
    assert polo_blocks
    assert "129000" in canonicals("\n".join(polo_blocks))
    assert "116000" in canonicals("\n".join(polo_blocks))

    yaris = search_passages.invoke({"query": "צריכת דלק", "files": ["yaris-mazda2.md"]})
    yaris_blocks = [body for file, _, body in blocks(yaris) if file == "yaris-mazda2.md"]
    assert {"20.6", "14.5"} <= canonicals("\n".join(yaris_blocks))


def test_compare_cites_each_file():
    text = compare.invoke({"files": ["kodiaq.md", "outlander.md"], "aspect": "נפח תא מטען"})
    assert "COMPARE kodiaq.md hit" in text
    assert "COMPARE outlander.md hit" in text
    by_file = {}
    current = None
    for line in text.splitlines():
        if line.startswith("COMPARE "):
            _, current, _ = line.split()
            by_file.setdefault(current, [])
        elif current:
            by_file[current].append(line)
    assert "289" in canonicals("\n".join(by_file["kodiaq.md"]))
    assert "148" in canonicals("\n".join(by_file["outlander.md"]))
    assert blocks("\n".join(by_file["kodiaq.md"]))
    assert blocks("\n".join(by_file["outlander.md"]))


def test_find_cars_cites_a_line_that_contains_the_price():
    text = find_cars.invoke({"make": "Volkswagen", "model": "Polo"})
    assert blocks(text)
    article = (ARTICLES / "polo.md").read_text(encoding="utf-8").splitlines()
    for file, lines, body in blocks(text):
        assert file == "polo.md"
        start = int(lines.split("-")[0])
        assert canonicals(body) <= canonicals(article[start - 1])


def test_mentions_are_not_returned_as_subjects():
    explorer = find_cars.invoke({"make": "Ford", "model": "Explorer"})
    assert explorer.startswith("Not reviewed. Mentioned in:")
    assert "palisade (as a rival)" in explorer
    assert "Search palisade.md" in explorer
    assert "[source:" not in explorer
    model_y = find_cars.invoke({"make": "Tesla", "model": "Model Y"})
    assert not model_y.startswith("[source:"), model_y
    for make, model in NOT_REVIEWED:
        text = find_cars.invoke({"make": make, "model": model})
        assert not text.startswith("[source:"), f"{make} {model}: {text.splitlines()[0]}"


def test_alias_and_trim_match_subjects():
    mazda = find_cars.invoke({"model": "mazda2"})
    assert "yaris-mazda2.md" in mazda
    assert mazda.startswith("[source:")
    sportage = find_cars.invoke({"model": "Sportage"})
    assert "tucson-sportage.md" in sportage
    assert "palisade.md" in find_cars.invoke({"model": "פליסייד"})
    assert find_cars.invoke({"model": "g7"}).startswith('I couldn\'t find "g7" by that name.')
    assert not find_cars.invoke({"model": "polo shirt"}).startswith("[source:")
    # palisade.md:36 Excellence is 350,000 and 358,000. :40 Calligraphy is 390,000 and 411,000.
    trim = find_cars.invoke({"model": "Calligraphy"})
    assert "palisade.md" in trim
    assert _prices(trim) == {"390000", "411000"}
    assert _prices(find_cars.invoke({"model": "קליגרפי"})) == {"390000", "411000"}
    assert _prices(find_cars.invoke({"model": "Excellence"})) == {"350000", "358000"}
    assert _prices(find_cars.invoke({"model": "Palisade"})) == {"350000", "358000", "390000", "411000"}


def test_search_gate():
    blocked = search_passages.invoke({"query": "מחיר", "car": "Ford Explorer"})
    assert blocked.startswith("Not reviewed. Mentioned in:")
    assert "[source:" not in blocked
    general = search_passages.invoke({"query": "נפח תא מטען"})
    assert "[source:" in general
    named = search_passages.invoke({"query": "Ford Explorer price"})
    assert "[source:" in named
    compared = search_passages.invoke({"query": "תא מטען", "car": "פאליסייד"})
    assert "palisade.md" in compared
    assert "explorer.md" not in compared
    wrong_file = compare.invoke({"files": ["polo.md"], "aspect": "מחיר", "car": "Ford Explorer"})
    assert wrong_file.startswith("Not reviewed.")
    mention_file = compare.invoke({"files": ["palisade.md"], "aspect": "Ford Explorer"})
    assert "COMPARE palisade.md hit" in mention_file


if __name__ == "__main__":
    test_search_cites_the_passage_that_holds_the_fact()
    test_compare_cites_each_file()
    test_find_cars_cites_a_line_that_contains_the_price()
    test_mentions_are_not_returned_as_subjects()
    test_alias_and_trim_match_subjects()
    test_search_gate()
    print("tools ok")
