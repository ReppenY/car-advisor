from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from car_advisor.paths import ROOT
from car_advisor.retrieval import format_hit, search
from car_advisor.store import classify, connect, gate, same_file


class FindInput(BaseModel):
    make: Optional[str] = Field(default=None, description="Make, or part of it")
    model: Optional[str] = Field(default=None, description="Model, or part of it")
    fuel: Optional[str] = Field(default=None, description="petrol, hybrid, plugin, electric, diesel")
    segment: Optional[str] = Field(default=None, description="city, family, suv, large-suv")
    seats_min: Optional[int] = None
    price_max: Optional[int] = Field(default=None, description="Maximum price in shekels")
    limit: int = Field(default=8, le=8)


class SearchInput(BaseModel):
    query: str = Field(description="What to look up in the road tests, e.g. trunk space")
    car: Optional[str] = Field(default=None, description="The reviewed car this question is about. One car. Omit when the question names no car.")
    files: Optional[list[str]] = Field(default=None, description="Optional file names, e.g. kodiaq.md")


class CompareInput(BaseModel):
    files: list[str] = Field(min_length=1, description="File name of each car, from find_cars")
    aspect: str = Field(description="The one thing to compare, e.g. ride comfort")
    car: Optional[str] = Field(default=None, description="Set only when looking inside one reviewed car's file. Omit when comparing two reviewed cars.")


def _car_line(row) -> str:
    trim = row["trim_en"] or row["trim_he"] or ""
    name = f"{row['make']} {row['model']}"
    if trim:
        name = f"{name} {trim}"
    price = f"{row['price_nis']:,} NIS" if row["price_nis"] else "price unknown"
    seats = f"{row['seats']} seats" if row["seats"] else "seats unknown"
    line = row["source_line"] or 1
    return (
        f"[source: {row['file']} lines {line}-{line}]\n"
        f"{name} | {row['segment']} | {row['fuel']} | {seats} | {price} | file={row['file']}"
    )


def _limit_files(files: Optional[list[str]], allowed: Optional[list[str]]) -> Optional[list[str]]:
    if allowed is None:
        return files
    if not files:
        return allowed
    chosen = [file for file in files if any(same_file(file, item) for item in allowed)]
    return chosen or allowed


@tool(args_schema=FindInput)
def find_cars(make=None, model=None, fuel=None, segment=None, seats_min=None, price_max=None, limit=8) -> str:
    """Find reviewed cars. Returns at most 8, each with a file name. A rival mentioned in a review is not a match."""
    query = " ".join(part for part in (make, model) if part)
    conn = connect(ROOT)
    try:
        status, rows, note = classify(
            conn, query, fuel=fuel, segment=segment,
            seats_min=seats_min, price_max=price_max, limit=limit,
        )
    finally:
        conn.close()
    if status == "rows":
        body = "\n\n".join(_car_line(row) for row in rows) if rows else "No matching cars."
        return f"{note}\n\n{body}" if note else body
    if status in {"mention", "candidates", "none"}:
        return note
    return "No matching cars."


@tool(args_schema=SearchInput)
def search_passages(query: str, car: Optional[str] = None, files: Optional[list[str]] = None) -> str:
    """Find what the road tests say about a topic. Pass car when the question names one reviewed car."""
    conn = connect(ROOT)
    try:
        blocked, allowed = gate(conn, car)
        if blocked:
            return blocked
        files = _limit_files(files, allowed)
    finally:
        conn.close()
    hits = search(query, k=4, files=files)
    if not hits:
        return "No passages found."
    return "\n\n".join(format_hit(hit) for hit in hits)


@tool(args_schema=CompareInput)
def compare(files: list[str], aspect: str, car: Optional[str] = None) -> str:
    """Look up one aspect in each file separately, so every car is represented."""
    conn = connect(ROOT)
    try:
        blocked, allowed = gate(conn, car)
        if blocked:
            return blocked
        files = _limit_files(files, allowed) or files
    finally:
        conn.close()
    blocks = []
    for file in files:
        hits = search(aspect, k=2, files=[file])
        status = "hit" if hits else "none"
        body = "\n\n".join(format_hit(hit) for hit in hits) if hits else "nothing on this aspect"
        blocks.append(f"COMPARE {file} {status}\n{body}")
    return "\n\n".join(blocks)


TOOLS = [find_cars, search_passages, compare]
TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}
