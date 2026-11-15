"""Edit distance, alias resolution, and the duplicate-family flag. No model."""

import io
from contextlib import redirect_stderr

from car_advisor.resolve import (
    allowed_distance,
    alias_records,
    compact,
    damerau_levenshtein,
    match_aliases,
    log_match,
    name_in_title,
    possible_duplicates,
)

SPORTAGE = [
    ("sportage", "kia-sportage"),
    ("kiasportage", "kia-sportage"),
    ("פאליסייד", "hyundai-palisade"),
    ("hyundaipalisade", "hyundai-palisade"),
    ("קליגרפי", "hyundai-palisade"),
    ("calligraphy", "hyundai-palisade"),
    ("g6", "xpeng-g6"),
    ("polo", "vw-polo"),
    ("volkswagenpolo", "vw-polo"),
    ("מאזדה2", "mazda-2"),
    ("mazda2", "mazda-2"),
]


def test_compact_hebrew():
    assert compact("פּוֹלוֹ") == compact("פולו")
    assert compact("ספורטאז'") == compact("ספורטאז")
    assert compact("ספורטאז׳") == compact("ספורטאז״")
    assert compact("סוף") == compact("סופ")
    assert compact("מאזדה 2") == "מאזדה2"
    assert compact("Mazda2") == compact("Mazda 2")


def test_distance_examples():
    assert damerau_levenshtein("sportige", "sportage") == 1
    assert damerau_levenshtein("sportgae", "sportage") == 1
    assert damerau_levenshtein(compact("פליסייד"), compact("פאליסייד")) == 1
    assert damerau_levenshtein("g6", "g7") == 1
    assert damerau_levenshtein("ca", "ac") == 1
    assert damerau_levenshtein("kitten", "sitting") == 3
    assert damerau_levenshtein("g7", "xpengg6") == 6
    assert allowed_distance(2) == 0
    assert allowed_distance(3) == 0
    assert allowed_distance(4) == 1
    assert allowed_distance(7) == 1
    assert allowed_distance(8) == 2


def test_query_resolution():
    assert match_aliases("sportige", SPORTAGE).family_ids == ["kia-sportage"]
    assert match_aliases("sportgae", SPORTAGE).family_ids == ["kia-sportage"]
    assert match_aliases("פליסייד", SPORTAGE).family_ids == ["hyundai-palisade"]
    assert match_aliases("mazda2", SPORTAGE).family_ids == ["mazda-2"]
    assert match_aliases("מאזדה 2", SPORTAGE).family_ids == ["mazda-2"]
    assert match_aliases("קליגרפי", SPORTAGE).family_ids == ["hyundai-palisade"]
    g7 = match_aliases("g7", SPORTAGE)
    assert g7.status == "none"
    assert g7.family_ids == []
    assert any(key == "g6" and distance == 1 for key, _family, distance in g7.near)
    shirt = match_aliases("polo shirt", SPORTAGE)
    assert shirt.status == "none"
    assert shirt.family_ids == []
    short = match_aliases("פול", [("פולו", "vw-polo")])
    assert short.status == "none"
    assert len(compact("פול")) <= 3
    assert match_aliases("פולי", [("פולו", "vw-polo")]).family_ids == ["vw-polo"]


def test_two_families_inside_the_distance_are_candidates():
    aliases = [("sportage", "kia"), ("sportige", "other")]
    found = match_aliases("sportxge", aliases)
    assert found.status == "candidates"
    assert set(found.family_ids) == {"kia", "other"}


def test_exact_alias_shared_by_two_families_is_ambiguous():
    aliases = [("ace", "one"), ("ace", "two"), ("alfaace", "one")]
    found = match_aliases("ace", aliases)
    assert found.status == "candidates"
    assert set(found.family_ids) == {"one", "two"}


def test_near_miss_and_unresolved_are_logged():
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        log_match("g7", match_aliases("g7", SPORTAGE))
        log_match("qwertyuiop", match_aliases("qwertyuiop", SPORTAGE))
    text = buffer.getvalue()
    assert '"event": "near_miss"' in text
    assert '"event": "unresolved"' in text
    assert "g7" in text
    assert "qwertyuiop" in text


def test_ingestion_flags_lookalikes_and_not_g6_g7():
    labels = [
        ("palisade", compact("פאליסייד")),
        ("typo", compact("פליסייד")),
        ("g6", "g6"),
        ("g7", "g7"),
    ]
    pairs = {(left, right) for left, right, _distance in possible_duplicates(labels)}
    assert ("palisade", "typo") in pairs
    assert ("g6", "g7") not in pairs


def test_aliases_come_from_the_catalog():
    cars = [
        {"family_id": "mazda2", "make": "Mazda", "model": "2", "name_he": "מאזדה 2", "name_en": "Mazda 2", "trim_he": "דיינמיק", "trim_en": "Dynamic"},
        {"family_id": "kia", "make": "Kia", "model": "Sportage", "name_he": "ספורטאז'", "name_en": "Kia Sportage", "trim_he": "", "trim_en": ""},
        {"family_id": "other", "make": "Other", "model": "Sportage", "name_he": "", "name_en": "Other Sportage", "trim_he": "", "trim_en": ""},
    ]
    rows = alias_records(cars)
    keys = {(key, family) for key, family, _kind in rows}
    assert ("mazda2", "mazda2") in keys
    assert (compact("מאזדה 2"), "mazda2") in keys
    assert (compact("דיינמיק"), "mazda2") in keys
    palisade = {"family_id": "palisade", "make": "Hyundai", "model": "Palisade", "name_he": "יונדאי פאליסייד", "name_en": "Hyundai Palisade", "trim_he": "", "trim_en": ""}
    assert (compact("פאליסייד"), "palisade") in {(key, family) for key, family, _kind in alias_records([palisade])}
    sportage = {family for key, family, kind in rows if key == "sportage" and kind == "model"}
    assert sportage == {"kia", "other"}


def test_heading_check_is_exact():
    palisade = "source: https://www.auto.co.il/hyundai-palisade/\n\n# יונדאי פאליסייד – מבחן דרכים\n"
    assert name_in_title("יונדאי פאליסייד", "Hyundai Palisade", palisade)
    assert not name_in_title("פורד אקספלורר", "Ford Explorer", palisade)
    geely = "source: https://www.auto.co.il/138201/\n\n# ג'ילי EX5 - מבחן דרכים\n"
    assert name_in_title("ג'ילי EX5", "Geely EX5", geely)
    assert not name_in_title("ג'ילי גאומטרי C", "Geely Geometry C", geely)


if __name__ == "__main__":
    test_compact_hebrew()
    test_distance_examples()
    test_query_resolution()
    test_two_families_inside_the_distance_are_candidates()
    test_exact_alias_shared_by_two_families_is_ambiguous()
    test_near_miss_and_unresolved_are_logged()
    test_ingestion_flags_lookalikes_and_not_g6_g7()
    test_aliases_come_from_the_catalog()
    test_heading_check_is_exact()
    print("resolve ok")
