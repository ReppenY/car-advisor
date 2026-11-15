import os
import re
import uuid
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from car_advisor.articles import ARTICLES
from car_advisor.llm import model
from car_advisor.log import log
from car_advisor.numbers import find_mentions
from car_advisor.paths import ARTICLES as ARTICLE_DIR
from car_advisor.resolve import compact, english_parts, name_in_title
from car_advisor.store import connect, rebuild_aliases

load_dotenv(override=True)


class VariantDraft(BaseModel):
    trim_he: str = ""
    trim_en: str = ""
    engine: str = ""
    segment: str = Field(default="", description="city, family, suv, large-suv, or other")
    fuel: str = Field(default="", description="petrol, hybrid, plugin, electric, diesel, or other")
    seats: int | None = None
    price_nis: int | None = Field(default=None, description="Price in shekels, digits only")


class CarDraft(BaseModel):
    name_he: str = Field(description="Hebrew make and model, no trim")
    name_en: str = Field(description="English make and model, no trim. Hyundai Palisade, Mazda 2, Xpeng G6, Geely EX5")
    role: Literal["subject", "mentioned"] = Field(
        description="subject: this article reviews that car. mentioned: any other car, including a rival"
    )
    variants: list[VariantDraft] = Field(default_factory=list)


class CarList(BaseModel):
    cars: list[CarDraft]


_SEAT_WORDS = {
    "שניים": 2, "שתיים": 2, "שלושה": 3, "שלוש": 3, "ארבעה": 4, "ארבע": 4,
    "חמישה": 5, "חמש": 5, "שישה": 6, "שש": 6, "שבעה": 7, "שבע": 7, "שמונה": 8,
}
_SEAT = re.compile(
    r"(?P<n>\d+|שניים|שתיים|שלושה|שלוש|ארבעה|ארבע|חמישה|חמש|שישה|שש|שבעה|שבע|שמונה)"
    r"\s*(?:מושבים|נוסעים|מקומות)"
)


def _seat_value(match) -> int | None:
    raw = match.group("n")
    value = int(raw) if raw.isdigit() else _SEAT_WORDS[raw]
    return value if 2 <= value <= 9 else None


def seats_near_price(text: str, price: int | None) -> int | None:
    """Seat count written next to this price. מושבי קפטן is not a seat count."""
    if price is None or not text:
        return None
    target = str(price)
    prices = []
    for mention in find_mentions(text):
        if not mention.canonical.isdigit() or int(mention.canonical) < 10000:
            continue
        at = text.find(mention.surface)
        if at >= 0:
            prices.append((at, mention.canonical))
    targets = [at for at, canonical in prices if canonical == target]
    if not targets:
        return None
    at = targets[0]
    seats = []
    for match in _SEAT.finditer(text):
        value = _seat_value(match)
        if value is None:
            continue
        seats.append((match.start(), value))
    owned = []
    for seat_at, value in seats:
        nearest = min(prices, key=lambda item: abs(item[0] - seat_at))
        if nearest[1] == target and nearest[0] == at:
            owned.append((abs(seat_at - at), value))
    return min(owned)[1] if owned else None


def _price_line(text: str, price: int | None) -> int | None:
    if price is None:
        return None
    target = str(price)
    for mention in find_mentions(text):
        if mention.canonical == target:
            return text[:text.find(mention.surface)].count("\n") + 1
    return None


def assign_role(role: str, name_he: str, name_en: str, article: str, slug: str) -> str:
    """A subject name has to appear in the title. No fuzzy match."""
    if name_in_title(name_he, name_en, article):
        if role != "subject":
            log("subject_promote", slug=slug, name=name_en)
        return "subject"
    if role == "subject":
        log("subject_downgrade", slug=slug, name=name_en, name_he=name_he)
    return "mentioned"


def _collect(draft: CarList, text: str, slug: str) -> tuple[list[dict], list[str]]:
    rows, problems, seen = [], [], set()
    for car in draft.cars:
        make, model = english_parts(car.name_en)
        if not make or not model:
            problems.append(f"missing English name for {car.name_he}")
            continue
        family_id = compact(f"{make} {model}")
        role = assign_role(car.role, car.name_he, f"{make} {model}", text, slug)
        variants = car.variants or [VariantDraft()]
        for variant in variants:
            line = _price_line(text, variant.price_nis)
            seats = variant.seats if variant.seats is not None else seats_near_price(text, variant.price_nis)
            if variant.price_nis is not None and line is None:
                label = variant.trim_en or variant.trim_he or model
                problems.append(f"{make} {label}: {variant.price_nis} is not in the article")
                continue
            key = (family_id, compact(variant.trim_en), compact(variant.trim_he), variant.price_nis, role)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "id": uuid.uuid4().hex[:12],
                "slug": slug,
                "make": make,
                "model": model,
                "segment": variant.segment,
                "fuel": variant.fuel,
                "seats": seats,
                "price_nis": variant.price_nis,
                "source_line": line,
                "file": f"{slug}.md",
                "role": role,
                "family_id": family_id,
                "name_he": car.name_he,
                "name_en": f"{make} {model}",
                "trim_he": variant.trim_he,
                "trim_en": variant.trim_en,
                "engine": variant.engine,
            })
    return rows, problems


def extract_article(slug: str, text: str) -> list[dict]:
    extractor = model().with_structured_output(CarList)
    prompt = (
        "List every car this road test names.\n"
        "name_en is the make and model only, with no trim: Hyundai Palisade, Mazda 2, Xpeng G6, Geely EX5.\n"
        "name_he is the Hebrew make and model of that same car, also with no trim.\n"
        "role=subject when this article reviews that car. role=mentioned for every other car, including rivals.\n"
        "Put each priced trim in variants. trim_en and trim_he are the trim (Calligraphy, קליגרפי), not the model.\n"
        "engine, fuel, seats, segment, and price_nis belong on the variant.\n"
        "seats is the passenger count written for that variant, digits or a Hebrew number: "
        "ל-7 מושבים is 7, לשמונה נוסעים is 8. Leave seats empty only when the article never states it.\n"
        "Copy prices from the text. Omit a price that is not written in the article.\n\n"
        + text
    )
    draft = extractor.invoke(prompt)
    rows, problems = _collect(draft, text, slug)
    if problems:
        log("extract_reject", slug=slug, problems=problems)
        draft = extractor.invoke(prompt + "\n\nRejected:\n" + "\n".join(problems))
        rows, _problems = _collect(draft, text, slug)
    return rows


def save_extracted(conn, slug: str, rows: list[dict]) -> None:
    conn.execute("DELETE FROM cars WHERE slug = ?", (slug,))
    for row in rows:
        conn.execute(
            """INSERT INTO cars (
                id, slug, make, model, segment, fuel, seats, price_nis, source_line, file, role,
                family_id, name_he, name_en, trim_he, trim_en, engine
            ) VALUES (
                :id, :slug, :make, :model, :segment, :fuel, :seats, :price_nis, :source_line, :file, :role,
                :family_id, :name_he, :name_en, :trim_he, :trim_en, :engine
            )""",
            row,
        )
    conn.commit()


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    conn = connect()
    try:
        for slug, _url in ARTICLES:
            text = (ARTICLE_DIR / f"{slug}.md").read_text(encoding="utf-8")
            rows = extract_article(slug, text)
            save_extracted(conn, slug, rows)
            subjects = sum(row["role"] == "subject" for row in rows)
            log("extract", slug=slug, cars=len(rows), subjects=subjects)
        rebuild_aliases(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
