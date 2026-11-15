"""Preference memory. No model calls."""

from car_advisor.graph import _segment_named, named_subject
from car_advisor.profile import (
    apply_update,
    hard_filters,
    mismatch_mentions,
    note_outside,
    preference_context,
    record_openness,
)
from car_advisor.store import connect


def _body(value="suv", strength="soft", turn=1, quote="אני מחפש רכב פנאי"):
    return {"body": {"value": value, "quote": quote, "strength": strength, "turn": turn}}


def test_later_hard_overrides_soft_and_soft_does_not_override_hard():
    profile = {}
    apply_update(profile, "body", "suv", "מחפש רכב פנאי", "soft", 1)
    apply_update(profile, "body", "city", "אולי עירונית", "soft", 2)
    assert profile["body"]["value"] == "suv"
    apply_update(profile, "body", "city", "רק עירונית", "hard", 3)
    assert profile["body"]["value"] == "city"
    assert profile["body"]["strength"] == "hard"
    apply_update(profile, "body", "suv", "מחפש רכב פנאי", "soft", 4)
    assert profile["body"]["value"] == "city"
    assert profile["body"]["strength"] == "hard"
    apply_update(profile, "body", "suv", "מחפש רכב פנאי", "hard", 5)
    assert profile["body"]["strength"] == "hard"
    assert profile["body"]["value"] == "city"


def test_an_explicit_change_of_mind_replaces_a_soft_preference():
    profile = _body()
    apply_update(profile, "body", "city", "בעצם עירונית", "soft", 2)
    assert profile["body"]["value"] == "city"


def test_two_city_questions_broaden_a_soft_suv_preference():
    profile = _body()
    counts = {}
    profile, counts = note_outside(profile, "city", counts)
    assert profile["body"]["value"] == "suv"
    assert counts["body:suv"] == 1
    profile, counts = note_outside(profile, "city", counts)
    assert profile["body"]["value"] == "suv; also considering city"
    profile, counts = note_outside(profile, "city", counts)
    assert counts["body:suv"] == 2


def test_direct_turns_hide_the_profile_and_a_mismatch_is_once():
    profile = _body()
    counts = {"body:suv": 1}
    text, pending, ask = preference_context("direct", profile, {}, False, counts)
    assert "מחפש" not in text
    assert "turn 1" not in text
    assert pending == []
    assert ask is False
    text, pending, ask = preference_context("recommendation", profile, {}, False, counts)
    assert "suv" in text
    assert pending == ["body"]
    assert ask is True
    text, pending, ask = preference_context("recommendation", profile, {"body": True}, True, counts)
    assert pending == []
    assert ask is False
    assert "already said" in text


def test_a_broadened_preference_is_recommended_without_another_question():
    profile = _body("suv; also considering city")
    text, pending, ask = preference_context(
        "recommendation", profile, {}, False, {"body:suv": 2}
    )
    assert "also considering city" in text
    assert pending == []
    assert ask is False
    assert "do not ask" in text.casefold()


def test_hard_constraints_filter_and_soft_ones_do_not():
    assert hard_filters(_body(strength="soft")) == {}
    assert hard_filters(_body(strength="hard")) == {"segment": "suv"}
    assert hard_filters(_body("suv; also considering city", strength="hard")) == {}
    budget = {"budget": {"value": "עד 150,000", "strength": "hard", "turn": 1, "quote": "עד 150,000"}}
    assert hard_filters(budget)["price_max"] == 150000


def test_an_openness_answer_is_recorded_once():
    profile = _body()
    profile, answered = record_openness(profile, "כן, פתוח", asked=True, answered=False)
    assert answered is True
    assert "also considering other types" in profile["body"]["value"]
    kept, answered = record_openness(profile, "לא פתוח", asked=True, answered=True)
    assert kept["body"]["strength"] == "soft"
    profile, answered = record_openness(_body(), "לא פתוח", asked=True, answered=False)
    assert profile["body"]["strength"] == "hard"


def test_mismatch_mentions_count_assistant_replies():
    transcript = (
        "Assistant: הפולו עולה 129,000.\n"
        "Assistant: זה לא רכב פנאי.\n"
        "User: מה עם מאזדה\n"
        "Assistant: This isn't an SUV.\n"
    )
    assert mismatch_mentions(transcript) == 2


def test_polo_mazda_and_yaris_are_outside_an_suv_preference():
    conn = connect()
    try:
        assert named_subject(conn, "ומה עם היאריס?")
        assert _segment_named(conn, "כמה עולה הפולו שנבחן?") == "city"
        assert _segment_named(conn, "וכמה עולה מאזדה 2?") == "city"
        assert _segment_named(conn, "ומה עם היאריס?") == "city"
        assert _segment_named(conn, "אני מחפש רכב פנאי.") is None
    finally:
        conn.close()
