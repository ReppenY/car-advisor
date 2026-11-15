"""Tool behavior on the extracted catalog. No model calls."""

from car_advisor.test_tools import (
    blocks,
    test_alias_and_trim_match_subjects,
    test_compare_cites_each_file,
    test_find_cars_cites_a_line_that_contains_the_price,
    test_mentions_are_not_returned_as_subjects,
    test_search_cites_the_passage_that_holds_the_fact,
    test_search_gate,
)
from car_advisor.tools import compare, find_cars, search_passages

__all__ = [
    "test_alias_and_trim_match_subjects",
    "test_compare_cites_each_file",
    "test_find_cars_cites_a_line_that_contains_the_price",
    "test_mentions_are_not_returned_as_subjects",
    "test_search_cites_the_passage_that_holds_the_fact",
    "test_search_gate",
]


def test_subject_car_limits_the_search_to_that_family():
    text = search_passages.invoke({"query": "מחיר", "car": "Volkswagen Polo"})
    assert {file for file, _lines, _body in blocks(text)} == {"polo.md"}
    compared = compare.invoke({"files": ["polo.md", "palisade.md"], "aspect": "מחיר", "car": "Hyundai Palisade"})
    assert "COMPARE palisade.md" in compared
    assert "COMPARE polo.md" not in compared


def test_mention_unknown_and_ambiguous_do_not_search():
    mention = search_passages.invoke({"query": "מחיר", "car": "Ford Explorer"})
    assert mention.startswith("Not reviewed. Mentioned in: palisade (as a rival).")
    assert "[source:" not in mention
    unknown = search_passages.invoke({"query": "מחיר", "car": "Mazda CX-5"})
    assert unknown.startswith('I couldn\'t find "Mazda CX-5" by that name.')
    assert "[source:" not in unknown
    ambiguous = search_passages.invoke({"query": "מחיר", "car": "Style"})
    assert ambiguous.startswith("Which car do you mean?")
    assert "[source:" not in ambiguous
    same = compare.invoke({"files": ["palisade.md"], "aspect": "מחיר", "car": "Style"})
    assert same.startswith("Which car do you mean?")


def test_no_car_argument_still_searches():
    text = search_passages.invoke({"query": "נפח תא מטען"})
    assert "[source:" in text
    live = find_cars.invoke({"make": "Tesla", "model": "Model Y"})
    assert live.startswith("Not reviewed. Mentioned in: xpeng-g6 (as a rival).")
