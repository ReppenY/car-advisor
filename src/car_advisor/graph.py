import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError
from typing_extensions import TypedDict

from car_advisor.llm import model
from car_advisor.log import log
from car_advisor.numbers import covers, find_mentions
from car_advisor.paths import CHECKPOINTS, ROOT
from car_advisor.profile import (
    apply_update,
    hard_filters,
    note_outside,
    preference_context,
    record_openness,
    remember_style,
    render_profile,
)
from car_advisor.prompts import RULES
from car_advisor.resolve import HEBREW_PREFIXES, match_aliases, strip_geresh
from car_advisor.schemas import FinalAnswer, Plan, Source, Triage
from car_advisor.store import _alias_pairs, connect, forms_for_sources, narrow_make
from car_advisor.tools import TOOLS, TOOLS_BY_NAME

MAX_STEPS = 6
TRIM_OVER = 300
KEEP_CHARS = 150
SOURCE = re.compile(r"\[source: (\S+) lines (\d+-\d+)\]")
RECOMMEND_CUE = ("ממליץ", "המלצ", "תמליץ", "המלצה", "recommend")
NAME = re.compile(r"[A-Za-z0-9]+|[א-ת]+")
SKIP = {
    "מה", "עם", "הוא", "היא", "זה", "את", "של", "על", "לא", "יש", "גם", "אז", "כי", "או",
    "אם", "כל", "עד", "בין", "יותר", "זול", "מחיר", "גרסה", "רכב", "פנאי", "עולה", "כמה",
    "שנבחן", "נבחן", "תן", "לי", "וכמה", "האם",
}

SYSTEM = RULES + " When you are done, call FinalAnswer. The answer text has no citations."


class State(TypedDict):
    messages: Annotated[list, add_messages]
    final: Optional[dict]
    steps: int
    plan: Optional[dict]
    profile: dict
    turn_kind: str
    mentioned_mismatch: dict
    asked_open: bool
    openness_answered: bool
    outside_counts: dict
    subject: Optional[dict]
    sources: list
    compare_required: list
    style: Optional[str]


def with_tool_results(messages):
    """A tool call with no result cannot be followed by the next user message."""
    repaired = []
    index = 0
    while index < len(messages):
        message = messages[index]
        repaired.append(message)
        calls = getattr(message, "tool_calls", None) or []
        index += 1
        if not calls:
            continue
        needed = [call["id"] for call in calls if call.get("id")]
        seen = set()
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            seen.add(messages[index].tool_call_id)
            repaired.append(messages[index])
            index += 1
        for call_id in needed:
            if call_id not in seen:
                repaired.append(ToolMessage(
                    content="That step did not finish. Answer the latest message.",
                    tool_call_id=call_id,
                    status="error",
                ))
    return repaired


def trimmed(messages):
    human_idx = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    current = human_idx[-1] if human_idx else 0
    out = []
    for i, message in enumerate(messages):
        if i < current and isinstance(message, ToolMessage) and len(str(message.content)) > TRIM_OVER:
            text = str(message.content)
            message = message.model_copy(update={
                "content": text[:KEEP_CHARS] + f"... [trimmed {len(text) - KEEP_CHARS} chars]"
            })
        out.append(message)
    return out


def user_texts(messages) -> list[str]:
    return [message.content for message in messages
            if isinstance(message, HumanMessage) and isinstance(message.content, str)]


def _latest_question(messages) -> str:
    texts = user_texts(messages)
    return texts[-1] if texts else ""


def _turn_index(messages) -> int:
    return len(user_texts(messages))


def _segment_of(conn, family_id: str) -> str | None:
    row = conn.execute(
        "SELECT segment FROM cars WHERE family_id = ? AND role = 'subject' AND segment IS NOT NULL AND segment != '' LIMIT 1",
        (family_id,),
    ).fetchone()
    return row["segment"] if row else None


def _segment_named(conn, text: str) -> str | None:
    found = scan_subject(conn, text)
    if not found:
        return None
    return _segment_of(conn, found["family_id"])


def triage(state: State):
    question = state["messages"][-1].content
    known = render_profile(state.get("profile") or {})
    conn = connect(ROOT)
    try:
        subject = scan_subject(conn, question)
    finally:
        conn.close()
    fallback = "direct" if subject else "recommendation"
    try:
        result = model().with_structured_output(Triage).invoke(
            "Classify this message and update the profile.\n"
            "kind=direct when the user asks about named cars. "
            "kind=recommendation when they want a suggestion or a list.\n"
            "strength=soft for a vague interest (looking for, מחפש). "
            "strength=hard for an explicit constraint (only, must, רק, חייב).\n"
            "A profile quote must be copied from the user's words. Leave profile_updates empty "
            "when the user revealed nothing new.\n\n"
            f"Known about this user:\n{known}\n\nMessage:\n{question}"
        )
    except Exception as error:
        log("triage_error", error=f"{type(error).__name__}: {error}")
        failed = {"plan": None, "turn_kind": fallback, "subject": subject}
        style = remember_style(state.get("style"), question)
        if style:
            failed["style"] = style
        return failed
    said = "\n".join(user_texts(state["messages"]))
    profile = dict(state.get("profile") or {})
    counts = dict(state.get("outside_counts") or {})
    asked = bool(state.get("asked_open"))
    answered = bool(state.get("openness_answered"))
    profile, answered = record_openness(profile, question, asked, answered)
    kept = []
    turn = _turn_index(state["messages"])
    for update in result.profile_updates:
        if update.quote and update.quote in said:
            apply_update(profile, update.field, update.value, update.quote, update.strength, turn)
            kept.append(update.field)
        else:
            log("profile_drop", field=update.field, quote=update.quote)
    if subject and not any(cue in question for cue in RECOMMEND_CUE):
        kind = "direct"
    else:
        kind = result.kind or fallback
    segment = None
    if kind == "direct" and subject:
        conn = connect(ROOT)
        try:
            segment = _segment_of(conn, subject["family_id"])
        finally:
            conn.close()
    if kind == "direct":
        profile, counts = note_outside(profile, segment, counts)
    plan = None
    if len(result.parts) >= 2:
        plan = Plan(steps=[{"id": i + 1, "goal": part} for i, part in enumerate(result.parts)]).model_dump()
    log("triage", parts=result.parts, kind=kind, profile=kept)
    result_state = {
        "plan": plan,
        "profile": profile,
        "turn_kind": kind,
        "outside_counts": counts,
        "openness_answered": answered,
        "subject": subject,
    }
    style = remember_style(state.get("style"), question)
    if style:
        result_state["style"] = style
    return result_state


def agent(state: State):
    plan = Plan(**state["plan"]) if state.get("plan") else None
    tools = TOOLS + [FinalAnswer] + ([Plan] if plan else [])
    choice = "FinalAnswer" if state["steps"] >= MAX_STEPS - 1 else "any"
    llm = model().bind_tools(tools, tool_choice=choice)
    messages = [SystemMessage(SYSTEM)] + with_tool_results(trimmed(state["messages"]))
    if plan:
        messages.append(HumanMessage(f"[Current plan]\n{plan.render()}"))
    note, _pending, _ask = preference_context(
        state.get("turn_kind") or "recommendation",
        state.get("profile") or {},
        state.get("mentioned_mismatch") or {},
        bool(state.get("asked_open")),
        state.get("outside_counts") or {},
    )
    shape = ""
    if state.get("style") == "short":
        shape = "\n\nThis session is short. Stay under half the usual length for this kind of answer."
    messages.append(HumanMessage(f"[Known about this user]\n{note}{shape}"))
    ai = llm.invoke(messages)
    log("llm", step=state["steps"], tools=[call["name"] for call in ai.tool_calls])
    return {"messages": [ai], "steps": state["steps"] + 1}


def _run_one(call):
    try:
        result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
    except Exception as error:
        result = f"Error: {type(error).__name__}: {error}"
    log("tool", name=call["name"], input=call["args"], output=str(result)[:200])
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def turn_tool_texts(messages, extra: list[str] | None = None) -> list[str]:
    last = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last = index
    texts = [str(message.content) for message in messages[last:] if isinstance(message, ToolMessage)]
    texts.extend(extra or [])
    return texts


def forms_in_texts(texts: list[str]) -> set[str]:
    allowed = set()
    for text in texts:
        for mention in find_mentions(text):
            allowed |= set(mention.forms)
    return allowed


def sources_for_answer(answer: str, texts: list[str]) -> list[dict]:
    """Source lines whose tool block contains a number the answer uses."""
    wanted = {mention.canonical for mention in find_mentions(answer)}
    if not wanted:
        return []
    found, seen = [], set()
    for text in texts:
        marks = list(SOURCE.finditer(text))
        for index, mark in enumerate(marks):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
            canons = {mention.canonical for mention in find_mentions(text[mark.start():end])}
            if not wanted & canons:
                continue
            item = (mark.group(1), mark.group(2))
            if item in seen:
                continue
            seen.add(item)
            found.append({"file": item[0], "lines": item[1]})
    return found


def _subject_key(conn, aliases, text: str):
    if not text or text in SKIP:
        return None
    found = match_aliases(text, aliases)
    if found.status != "one":
        return None
    family = found.family_ids[0]
    if found.kinds.get(family) == "make":
        return None
    row = conn.execute(
        "SELECT 1 FROM cars WHERE family_id = ? AND role = 'subject' LIMIT 1",
        (family,),
    ).fetchone()
    if not row:
        return None
    return found


def scan_subject(conn, text: str) -> dict | None:
    """One reviewed car in this message: its alias key and family. None otherwise."""
    tokens = NAME.findall(strip_geresh(text))
    aliases = _alias_pairs(conn)
    families = []
    keys = {}
    used = set()

    def take(piece: str) -> bool:
        options = [piece]
        if len(piece) > 3 and piece[0] in HEBREW_PREFIXES and "א" <= piece[0] <= "ת":
            options.append(piece[1:])
        for option in options:
            if option in SKIP:
                continue
            found = _subject_key(conn, aliases, option)
            if not found:
                continue
            family = found.family_ids[0]
            if family not in families:
                families.append(family)
                keys[family] = found.keys.get(family) or option
            return True
        return False

    for index in range(len(tokens) - 1):
        if take(f"{tokens[index]} {tokens[index + 1]}"):
            used.add(index)
            used.add(index + 1)
    for index, token in enumerate(tokens):
        if index not in used:
            take(token)
    if len(families) == 1:
        family = families[0]
        return {"key": keys[family], "family_id": family}
    if families:
        return None
    return _lone_make(conn, aliases, tokens)


def _lone_make(conn, aliases, tokens):
    """A message that names a make and nothing else. One reviewed model is that car."""
    names = [token for token in tokens if token not in SKIP]
    if len(names) != 1:
        return None
    piece = names[0]
    options = [piece]
    if len(piece) > 3 and piece[0] in HEBREW_PREFIXES and "א" <= piece[0] <= "ת":
        options.append(piece[1:])
    for option in options:
        found = narrow_make(conn, match_aliases(option, aliases))
        if found.status != "one" or not found.family_ids:
            continue
        family = found.family_ids[0]
        if found.kinds.get(family) != "make":
            continue
        row = conn.execute(
            "SELECT 1 FROM cars WHERE family_id = ? AND role = 'subject' LIMIT 1",
            (family,),
        ).fetchone()
        if not row:
            continue
        return {"key": option, "family_id": family}
    return None


def named_subject(conn, text: str) -> str | None:
    """One reviewed car named in this message. None when it names none, or more than one.

    A trim resolves to its family (Excellence is the Palisade). A name that does
    not resolve leaves the previous car alone.
    """
    found = scan_subject(conn, text)
    return found["key"] if found else None


def _allowed_forms(sources: list[dict], messages, profile: dict, texts: list[str] | None = None) -> set[str]:
    conn = connect(ROOT)
    try:
        allowed = forms_for_sources(conn, sources)
    finally:
        conn.close()
    for text in user_texts(messages):
        for mention in find_mentions(text):
            allowed |= set(mention.forms)
    allowed |= forms_in_texts(texts or [])
    for entry in (profile or {}).values():
        for mention in find_mentions(str(entry.get("value", ""))):
            allowed |= set(mention.forms)
    return allowed


def _check_answer(answer: FinalAnswer, state: State, sources: list[dict], required: set[str],
                  texts: list[str] | None = None) -> str | None:
    texts = texts or []
    emitted = {(source["file"], source["lines"]) for source in sources}
    for item in sources_for_answer(answer.answer, texts):
        if (item["file"], item["lines"]) in emitted and not any(
            source.file == item["file"] and source.lines == item["lines"] for source in answer.sources
        ):
            answer.sources.append(Source(**item))
    for source in answer.sources:
        if (source.file, source.lines) not in emitted:
            return f"{source.file} lines {source.lines} was not returned by a tool this turn."
    missing_numbers = covers(answer.answer, _allowed_forms(
        [source.model_dump() for source in answer.sources], state["messages"], state.get("profile") or {}, texts))
    if missing_numbers:
        return f"Cite the find_cars rows for these prices: {', '.join(missing_numbers)}."
    cited = {source.file for source in answer.sources}
    absent = sorted(required - cited)
    if absent:
        return f"The comparison is missing a passage from: {', '.join(absent)}."
    return None


def act(state: State):
    calls = state["messages"][-1].tool_calls
    results = []
    try:
        return _act(state, calls, results)
    except Exception as error:
        log("act_error", error=f"{type(error).__name__}: {error}")
        done = {message.tool_call_id for message in results if isinstance(message, ToolMessage)}
        for call in calls:
            if call.get("id") not in done:
                results.append(ToolMessage(
                    content=f"Error: {type(error).__name__}: {error}",
                    tool_call_id=call["id"],
                    status="error",
                ))
        return {"messages": results, "final": None}


def _act(state: State, calls, results):
    car = (state.get("subject") or {}).get("key")
    if state.get("turn_kind") == "recommendation" and not car:
        forced = hard_filters(state.get("profile") or {})
        for call in calls:
            if call["name"] == "find_cars":
                call["args"].update(forced)
    if car:
        for call in calls:
            if call["name"] == "find_cars":
                call["args"]["make"] = None
                call["args"]["model"] = car
            elif call["name"] in ("search_passages", "compare"):
                call["args"]["car"] = car
    normal = [call for call in calls if call["name"] not in ("FinalAnswer", "Plan")]
    plan_calls = [call for call in calls if call["name"] == "Plan"]
    submits = [call for call in calls if call["name"] == "FinalAnswer"]

    with ThreadPoolExecutor() as pool:
        results.extend(pool.map(_run_one, normal))

    sources = list(state.get("sources") or [])
    required = set(state.get("compare_required") or [])
    for call, message in zip(normal, results):
        for match in SOURCE.finditer(str(message.content)):
            sources.append({"file": match.group(1), "lines": match.group(2)})
        if call["name"] == "compare":
            for line in str(message.content).splitlines():
                parts = line.split()
                if len(parts) != 3 or parts[0] != "COMPARE":
                    continue
                _, file, status = parts
                if status == "hit":
                    required.add(file)
    plan = Plan(**state["plan"]) if state.get("plan") else None
    for call in plan_calls:
        try:
            plan = Plan.model_validate(call["args"])
            results.append(ToolMessage(content="Plan saved.", tool_call_id=call["id"]))
        except ValidationError as error:
            results.append(ToolMessage(content=f"Invalid plan: {error}", tool_call_id=call["id"], status="error"))

    final = None
    forced = state["steps"] >= MAX_STEPS
    for call in submits:
        if plan and plan.unfinished() and not forced:
            missing = ", ".join(f"{step.id}. {step.goal}" for step in plan.unfinished())
            results.append(ToolMessage(
                content=f"Your plan still has unfinished steps: {missing}. Finish them or mark them failed.",
                tool_call_id=call["id"], status="error"))
            continue
        try:
            answer = FinalAnswer.model_validate(call["args"])
        except ValidationError as error:
            results.append(ToolMessage(
                content=f"Invalid answer: {error}. Fix it and call FinalAnswer again.",
                tool_call_id=call["id"], status="error"))
            continue
        problem = _check_answer(
            answer, state, sources, required,
            turn_tool_texts(state["messages"], [str(message.content) for message in results]),
        )
        if problem:
            log("validate", ok=False, problem=problem)
            results.append(ToolMessage(
                content=f"Rejected: {problem} Fix the answer and call FinalAnswer again.",
                tool_call_id=call["id"], status="error"))
            continue
        final = answer.model_dump()
        results.append(ToolMessage(content="Answer accepted.", tool_call_id=call["id"]))
        log("validate", ok=True)

    mentioned = dict(state.get("mentioned_mismatch") or {})
    asked = bool(state.get("asked_open"))
    if final:
        _note, pending, ask = preference_context(
            state.get("turn_kind") or "recommendation",
            state.get("profile") or {},
            mentioned,
            asked,
            state.get("outside_counts") or {},
        )
        for field in pending:
            mentioned[field] = True
        if ask:
            asked = True
    return {
        "messages": results,
        "final": final,
        "plan": plan.model_dump() if plan else None,
        "sources": sources,
        "compare_required": sorted(required),
        "mentioned_mismatch": mentioned,
        "asked_open": asked,
    }


def after_agent(state: State):
    return "act" if state["messages"][-1].tool_calls else END


def after_act(state: State):
    return END if state["final"] else "agent"


builder = StateGraph(State)
builder.add_node("triage", triage)
builder.add_node("agent", agent)
builder.add_node("act", act)
builder.add_edge(START, "triage")
builder.add_edge("triage", "agent")
builder.add_conditional_edges("agent", after_agent, ["act", END])
builder.add_conditional_edges("act", after_act, ["agent", END])

CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
graph = builder.compile(checkpointer=SqliteSaver(sqlite3.connect(CHECKPOINTS, check_same_thread=False)))


def run_turn(session_id: str, user_text: str) -> FinalAnswer | None:
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 2 * MAX_STEPS + 4}
    turn = {"messages": [HumanMessage(user_text)], "final": None, "steps": 0,
            "plan": None, "sources": [], "compare_required": [], "subject": None}
    graph.invoke(turn, config=config)
    final = graph.get_state(config).values.get("final")
    return FinalAnswer(**final) if final else None
