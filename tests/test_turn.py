"""Follow-up name resolution, grounded tool numbers, and seat counts. No model calls."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from car_advisor.extract import seats_near_price
from car_advisor.graph import _check_answer, named_subject, with_tool_results
from car_advisor.schemas import FinalAnswer
from car_advisor.store import connect


def test_excellence_in_a_follow_up_resolves_to_the_palisade():
    conn = connect()
    try:
        assert named_subject(conn, "מה עם האקסלנס") == "אקסלנס"
        assert named_subject(conn, "כמה עולה הפולו שנבחן?")
        polo = named_subject(conn, "כמה עולה הפולו שנבחן?")
        assert polo in {"פולו", "polo"}
        assert named_subject(conn, "אני מחפש רכב פנאי") is None
        assert named_subject(conn, "ומה רשום שם על האקספלורר?") is None
    finally:
        conn.close()


def test_find_cars_prices_are_grounded_and_their_source_is_attached():
    tool = (
        "[source: polo.md lines 27-27]\n"
        "Volkswagen Polo Style Plus | city | petrol | seats unknown | 129,000 NIS | file=polo.md"
    )
    answer = FinalAnswer(answer="הפולו שנבחן עולה 129,000 שקלים.", sources=[])
    state = {"messages": [HumanMessage("כמה עולה הפולו?")], "profile": {}}
    problem = _check_answer(
        answer, state,
        [{"file": "polo.md", "lines": "27-27"}],
        set(),
        [tool],
    )
    assert problem is None
    assert answer.sources[0].file == "polo.md"
    assert answer.sources[0].lines == "27-27"


def test_a_number_from_nowhere_asks_for_the_find_cars_row():
    answer = FinalAnswer(answer="המחיר הוא 999,000 שקלים.", sources=[])
    state = {"messages": [HumanMessage("כמה זה עולה?")], "profile": {}}
    problem = _check_answer(answer, state, [], set(), ["No matching cars."])
    assert problem == "Cite the find_cars rows for these prices: 999,000 שקלים."


def test_a_finished_tool_call_stays_in_front_of_the_next_message():
    call = AIMessage(content="", tool_calls=[{
        "name": "FinalAnswer", "id": "toolu_missing", "args": {"answer": "היי"},
    }])
    follow = HumanMessage("looking for an SUV")
    repaired = with_tool_results([call, follow])
    assert isinstance(repaired[1], ToolMessage)
    assert repaired[1].tool_call_id == "toolu_missing"
    assert repaired[2] is follow
    done = ToolMessage(content="Answer accepted.", tool_call_id="toolu_ok")
    kept = with_tool_results([
        AIMessage(content="", tool_calls=[{"name": "FinalAnswer", "id": "toolu_ok", "args": {}}]),
        done,
        follow,
    ])
    assert kept[1] is done
    assert len(kept) == 3


def test_seat_count_is_the_one_next_to_the_price():
    line = (
        "אקסלנס בהנעה קדמית בלבד ולשמונה נוסעים, העולה 350,000 שקלים. "
        "במחיר 358,000 שקלים אותה רמת אבזור עם 2 מושבי קפטן, ל-7 מושבים."
    )
    assert seats_near_price(line, 350000) == 8
    assert seats_near_price(line, 358000) == 7
    calligraphy = "קליגרפי מוצעת רק עם 7 מושבים ב-390,000 שקלים ובהנעה כפולה ב-411,000 שקלים."
    assert seats_near_price(calligraphy, 390000) == 7
    assert seats_near_price("116,000 שקלים לגרסת סטייל.", 116000) is None
