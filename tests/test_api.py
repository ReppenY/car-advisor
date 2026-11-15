"""The page returns an answer and a review list. No model calls."""

from fastapi.testclient import TestClient

from car_advisor.api import app
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from car_advisor.present import conversation_turns, for_user
from car_advisor.schemas import FinalAnswer, Source


def test_the_person_does_not_see_tools_or_sources():
    raw = (
        "הפולו שנבחן עולה 129,000 שקלים.\n"
        "[source: polo.md lines 27-27]\n"
        "find_cars said so. See polo.md."
    )
    shown = for_user(raw)
    assert "129,000" in shown
    assert "source" not in shown
    assert "polo.md" not in shown
    assert "find_cars" not in shown


def test_chat_returns_only_the_answer(monkeypatch):
    def fake_turn(session_id, message):
        assert session_id == "s1"
        assert message == "כמה עולה הפולו?"
        return FinalAnswer(
            answer="129,000 שקלים. [source: polo.md lines 27-27]",
            sources=[Source(file="polo.md", lines="27-27")],
        )

    monkeypatch.setattr("car_advisor.api.run_turn", fake_turn)
    client = TestClient(app)
    response = client.post("/api/chat", json={"session_id": "s1", "message": "כמה עולה הפולו?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "sources"}
    assert "129,000" in body["answer"]
    assert "source" not in body["answer"]
    assert "polo.md" not in body["answer"]
    assert body["sources"] == "מקורות: polo.md:27-27"


def test_a_failed_turn_returns_json(monkeypatch):
    def boom(session_id, message):
        raise RuntimeError("tool_use without tool_result")

    monkeypatch.setattr("car_advisor.api.run_turn", boom)
    client = TestClient(app)
    response = client.post("/api/chat", json={"session_id": "s1", "message": "היי"})
    assert response.status_code == 502
    assert response.json()["detail"] == "That turn failed. Send the message again."


def test_a_saved_turn_keeps_the_accepted_answer_only():
    messages = [
        HumanMessage("היי"),
        AIMessage(content="", tool_calls=[{
            "name": "FinalAnswer", "id": "draft", "args": {"answer": "too long", "sources": []},
        }]),
        ToolMessage(content="Rejected: too long", tool_call_id="draft", status="error"),
        AIMessage(content="", tool_calls=[{
            "name": "FinalAnswer",
            "id": "ok",
            "args": {"answer": "היי.", "sources": [{"file": "polo.md", "lines": "1-1"}]},
        }]),
        ToolMessage(content="Answer accepted.", tool_call_id="ok"),
        HumanMessage("כמה עולה הפולו?"),
    ]
    turns = conversation_turns(messages)
    assert turns[0]["you"] == "היי"
    assert turns[0]["answer"] == "היי."
    assert turns[0]["sources"] == "מקורות: polo.md:1-1"
    assert turns[1]["answer"] == ""


def test_sessions_are_saved_chats_not_eval_runs():
    client = TestClient(app)
    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert all(not item["id"].startswith("eval-") for item in sessions)
    assert all(item["title"] for item in sessions)
    page = client.get("/")
    assert "New conversation" in page.text


def test_page_lists_review_titles_and_rejects_a_bad_url():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Add a review" in page.text
    assert "Thinking..." in page.text
    assert "body.sources" in page.text
    assert 'event.key !== "Enter"' in page.text
    assert "[source:" not in page.text
    listed = client.get("/api/articles")
    assert listed.status_code == 200
    titles = listed.json()["articles"]
    assert titles
    assert all(".md" not in title for title in titles)
    rejected = client.post("/api/articles", json={"url": "not a url"})
    assert rejected.status_code == 400
