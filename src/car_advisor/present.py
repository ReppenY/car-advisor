"""The words the person sees. Citations stay in the sources field."""

import re

from langchain_core.messages import HumanMessage, ToolMessage

_SOURCE = re.compile(r"\[source:[^\]]*\]")
_FILE = re.compile(r"\b[\w.-]+\.md\b")
_TOOL = re.compile(r"\b(?:find_cars|search_passages|compare|FinalAnswer)\b")


def sources_line(sources) -> str:
    parts = []
    for source in sources or []:
        file = source["file"] if isinstance(source, dict) else getattr(source, "file", "")
        lines = source["lines"] if isinstance(source, dict) else getattr(source, "lines", "")
        if file and lines:
            parts.append(f"{file}:{lines}")
    if not parts:
        return ""
    return "מקורות: " + ", ".join(parts)


def conversation_turns(messages) -> list[dict]:
    """User text and the accepted answer. Rejected drafts stay out."""
    turns = []
    you = None
    drafts: dict[str, dict] = {}
    answer = ""
    sources = ""

    def close():
        nonlocal you, drafts, answer, sources
        if you is None:
            return
        turns.append({"you": you, "answer": for_user(answer), "sources": sources})
        you = None
        drafts = {}
        answer = ""
        sources = ""

    for message in messages or []:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            close()
            you = message.content
            continue
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") == "FinalAnswer" and call.get("id"):
                drafts[call["id"]] = call.get("args") or {}
        if isinstance(message, ToolMessage) and message.content == "Answer accepted.":
            args = drafts.get(message.tool_call_id) or {}
            text = args.get("answer") or ""
            if text:
                answer = text
                sources = sources_line(args.get("sources") or [])
    close()
    return turns


def for_user(text: str) -> str:
    cleaned = _SOURCE.sub("", text or "")
    cleaned = _FILE.sub("", cleaned)
    cleaned = _TOOL.sub("", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    kept = [line for line in lines if line]
    return "\n".join(kept).strip()
