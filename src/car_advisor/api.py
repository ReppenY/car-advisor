"""Local page and the two calls it makes: chat, and add one article."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from car_advisor.graph import graph, run_turn
from car_advisor.ingest import add_url
from car_advisor.present import conversation_turns, for_user, sources_line
from car_advisor.store import connect

_HIDDEN = ("eval-", "ui-stream", "stream-", "dbg-", "case-")

PAGE = Path(__file__).resolve().parent / "static" / "index.html"
app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ArticleRequest(BaseModel):
    url: str


@app.get("/")
def page():
    return FileResponse(PAGE)


@app.get("/api/articles")
def articles():
    conn = connect()
    try:
        rows = conn.execute("SELECT title FROM articles ORDER BY title").fetchall()
    finally:
        conn.close()
    return {"articles": [row["title"] for row in rows if row["title"]]}


def _session_rows() -> list[tuple[str, str]]:
    saver = graph.checkpointer
    with saver.cursor(transaction=False) as cursor:
        cursor.execute(
            """
            SELECT thread_id, MAX(checkpoint_id) AS latest
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY latest DESC
            """
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]


def _visible(session_id: str) -> bool:
    return bool(session_id) and not session_id.startswith(_HIDDEN)


@app.get("/api/sessions")
def sessions():
    listed = []
    for session_id, _latest in _session_rows():
        if not _visible(session_id):
            continue
        state = graph.get_state({"configurable": {"thread_id": session_id}}).values
        turns = conversation_turns(state.get("messages") or [])
        if not turns:
            continue
        title = turns[0]["you"].strip().splitlines()[0]
        listed.append({"id": session_id, "title": title[:48]})
    return {"sessions": listed}


@app.get("/api/sessions/{session_id}")
def session(session_id: str):
    if not session_id.strip() or not _visible(session_id):
        raise HTTPException(404, "No such conversation.")
    state = graph.get_state({"configurable": {"thread_id": session_id}}).values
    return {"turns": conversation_turns(state.get("messages") or [])}


@app.post("/api/chat")
def chat(body: ChatRequest):
    session = body.session_id.strip()
    message = body.message.strip()
    if not session or not message:
        raise HTTPException(400, "A session and a message are required.")
    try:
        final = run_turn(session, message)
    except Exception as error:
        raise HTTPException(502, "That turn failed. Send the message again.") from error
    if final is None:
        raise HTTPException(502, "No answer this time.")
    return {"answer": for_user(final.answer), "sources": sources_line(final.sources)}


@app.post("/api/articles")
def article(body: ArticleRequest):
    try:
        return add_url(body.url)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError:
        raise HTTPException(400, "That page did not look like a road test.") from None


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
