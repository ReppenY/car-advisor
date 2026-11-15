import hashlib
import json
import sqlite3
from pathlib import Path

from car_advisor.log import log
from car_advisor.paths import CATALOG, ROOT
from car_advisor.resolve import (
    alias_records,
    compact,
    hebrew_make,
    log_match,
    match_aliases,
    possible_duplicates,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    slug TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    content_hash TEXT NOT NULL,
    embedded_model TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    heading TEXT,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS numbers (
    chunk_id TEXT NOT NULL,
    canonical TEXT NOT NULL,
    forms TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cars (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    make TEXT NOT NULL,
    model TEXT NOT NULL,
    segment TEXT,
    fuel TEXT,
    seats INTEGER,
    price_nis INTEGER,
    source_line INTEGER,
    file TEXT NOT NULL,
    role TEXT,
    family_id TEXT,
    name_he TEXT,
    name_en TEXT,
    trim_he TEXT,
    trim_en TEXT,
    engine TEXT
);
CREATE TABLE IF NOT EXISTS aliases (
    key TEXT NOT NULL,
    family_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (key, family_id, kind)
);
"""


def connect(root: Path | None = None) -> sqlite3.Connection:
    root = root or ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = root / "catalog.db" if root != ROOT else CATALOG
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(cars)")}
    added = False
    for name, kind in (
        ("role", "TEXT"), ("family_id", "TEXT"), ("name_he", "TEXT"), ("name_en", "TEXT"),
        ("trim_he", "TEXT"), ("trim_en", "TEXT"), ("engine", "TEXT"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE cars ADD COLUMN {name} {kind}")
            added = True
    if added:
        conn.commit()
    return conn


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_id(slug: str, text: str) -> str:
    return hashlib.sha256(f"{slug}\n{text}".encode()).hexdigest()[:24]


def article_row(conn, slug: str):
    return conn.execute("SELECT * FROM articles WHERE slug = ?", (slug,)).fetchone()


def replace_article(conn, slug: str, url: str, title: str, digest: str, model: str,
                    chunks: list[dict]):
    conn.execute("DELETE FROM numbers WHERE chunk_id IN (SELECT id FROM chunks WHERE slug = ?)", (slug,))
    conn.execute("DELETE FROM chunks WHERE slug = ?", (slug,))
    conn.execute(
        "INSERT OR REPLACE INTO articles (slug, url, title, content_hash, embedded_model) VALUES (?, ?, ?, ?, ?)",
        (slug, url, title, digest, model),
    )
    for chunk in chunks:
        conn.execute(
            "INSERT INTO chunks (id, slug, start_line, end_line, heading, text) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk["id"], slug, chunk["start_line"], chunk["end_line"], chunk["heading"], chunk["text"]),
        )
        for mention in chunk["mentions"]:
            conn.execute(
                "INSERT INTO numbers (chunk_id, canonical, forms) VALUES (?, ?, ?)",
                (chunk["id"], mention.canonical, json.dumps(sorted(mention.forms), ensure_ascii=False)),
            )
    conn.commit()


def all_chunks(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM chunks ORDER BY slug, start_line").fetchall()


def forms_for_sources(conn, sources: list[dict]) -> set[str]:
    allowed = set()
    for source in sources:
        slug = source["file"].removesuffix(".md")
        start, _, end = source["lines"].partition("-")
        rows = conn.execute(
            """SELECT numbers.forms FROM numbers
               JOIN chunks ON chunks.id = numbers.chunk_id
               WHERE chunks.slug = ? AND chunks.start_line <= ? AND chunks.end_line >= ?""",
            (slug, int(end or start), int(start)),
        )
        for row in rows:
            allowed.update(json.loads(row["forms"]))
    return allowed


def _filtered(conn, *, role, fuel=None, segment=None, seats_min=None, price_max=None):
    rows = conn.execute(
        "SELECT * FROM cars WHERE role = ? ORDER BY price_nis IS NULL, price_nis",
        (role,),
    ).fetchall()
    return [row for row in rows if _matches(row, fuel, segment, seats_min, price_max)]


def _matches(row, fuel, segment, seats_min, price_max) -> bool:
    if fuel and fuel.casefold() not in (row["fuel"] or "").casefold():
        return False
    if segment and segment.casefold() not in (row["segment"] or "").casefold():
        return False
    if seats_min is not None and (row["seats"] is None or row["seats"] < seats_min):
        return False
    if price_max is not None and (row["price_nis"] is None or row["price_nis"] > price_max):
        return False
    return True


def _alias_pairs(conn) -> list[tuple[str, str, str]]:
    return [
        (row["key"], row["family_id"], row["kind"])
        for row in conn.execute("SELECT key, family_id, kind FROM aliases")
    ]


def trim_rows(rows, found):
    """A trim name returns that trim's rows. The family name returns every variant."""
    if found.status != "one" or not found.family_ids:
        return rows
    family = found.family_ids[0]
    if found.kinds.get(family) != "trim":
        return rows
    key = found.keys.get(family, "")
    return [
        row for row in rows
        if key and key in {compact(row["trim_en"] or ""), compact(row["trim_he"] or "")}
    ]


def rival_note(slugs: list[str]) -> str:
    listed = ", ".join(f"{slug} (as a rival)" for slug in slugs)
    files = ", ".join(f"{slug}.md" for slug in slugs)
    return f"Not reviewed. Mentioned in: {listed}. Search {files} for what that review says."


def candidate_note(conn, family_ids: list[str]) -> str:
    lines = ["Which car do you mean?"]
    for family_id in family_ids:
        row = conn.execute(
            "SELECT make, model, file, role FROM cars WHERE family_id = ? ORDER BY role DESC LIMIT 1",
            (family_id,),
        ).fetchone()
        if row:
            lines.append(f"- {row['make']} {row['model']} ({row['file']}, {row['role']})")
    return "\n".join(lines)


def reviewed_choice(conn, family_ids: list[str]) -> str:
    lines = ["Which reviewed model do you mean?"]
    for family_id in family_ids:
        row = conn.execute(
            "SELECT make, model, file FROM cars WHERE family_id = ? AND role = 'subject' LIMIT 1",
            (family_id,),
        ).fetchone()
        if row:
            lines.append(f"- {row['make']} {row['model']} ({row['file']})")
    return "\n".join(lines)


def miss_note(query: str, labels: list[str]) -> str:
    text = f'I couldn\'t find "{query}" by that name.'
    if labels:
        text += " Close names, not a match: " + ", ".join(labels) + "."
    return text


def narrow_make(conn, found):
    """A make with one reviewed model becomes that model. Several stay a choice."""
    if not found.family_ids or any(found.kinds.get(family) != "make" for family in found.family_ids):
        return found
    marks = ",".join("?" * len(found.family_ids))
    rows = conn.execute(
        f"SELECT DISTINCT family_id FROM cars WHERE role = 'subject' AND family_id IN ({marks})",
        found.family_ids,
    ).fetchall()
    have = {row["family_id"] for row in rows}
    reviewed = [family for family in found.family_ids if family in have]
    if len(reviewed) == 1:
        family = reviewed[0]
        return type(found)(
            "one", [family], found.near,
            {family: "make"}, {family: found.keys.get(family, "")}, found.suggestions,
        )
    if len(reviewed) > 1:
        return type(found)(
            "candidates", reviewed, found.near,
            {family: "make" for family in reviewed},
            {family: found.keys.get(family, "") for family in reviewed},
            found.suggestions,
        )
    return found


def _suggestion_labels(conn, suggestions) -> list[str]:
    labels = []
    seen = set()
    for _alias, family, kind, _distance in suggestions:
        row = conn.execute(
            "SELECT make, model, name_he FROM cars WHERE family_id = ? ORDER BY role DESC LIMIT 1",
            (family,),
        ).fetchone()
        if not row:
            continue
        if kind == "make":
            he = hebrew_make(row["name_he"] or "")
            label = row["make"] or he
            if he and label and compact(he) != compact(label):
                label = f"{label} ({he})"
        else:
            label = f"{row['make']} {row['model']}".strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) == 5:
            break
    return labels


def classify(conn, query: str, *, fuel=None, segment=None, seats_min=None, price_max=None, limit=8):
    """status, payload, note. A make narrows to the reviewed models of that make."""
    cap = min(limit or 8, 8)
    if not compact(query or ""):
        rows = _filtered(conn, role="subject", fuel=fuel, segment=segment,
                         seats_min=seats_min, price_max=price_max)
        return "rows", rows[:cap], ""
    found = narrow_make(conn, match_aliases(query, _alias_pairs(conn)))
    log_match(query, found)
    if found.status == "none":
        return "none", [], miss_note(query, _suggestion_labels(conn, found.suggestions))
    if found.status == "candidates":
        if all(found.kinds.get(family) == "make" for family in found.family_ids):
            note = reviewed_choice(conn, found.family_ids)
        else:
            note = candidate_note(conn, found.family_ids)
        return "candidates", found.family_ids, note
    rows = conn.execute("SELECT * FROM cars WHERE family_id = ?", (found.family_ids[0],)).fetchall()
    subjects = [row for row in rows if row["role"] == "subject" and _matches(row, fuel, segment, seats_min, price_max)]
    subjects = trim_rows(subjects, found)
    if subjects:
        subjects.sort(key=lambda row: (row["price_nis"] is None, row["price_nis"] or 0))
        note = ""
        if found.kinds.get(found.family_ids[0]) == "make":
            row = subjects[0]
            note = f"We have the {row['make']} {row['model']} review."
        return "rows", subjects[:cap], note
    mentions = [row for row in rows if row["role"] == "mentioned"]
    if mentions:
        slugs = list(dict.fromkeys(row["slug"] for row in mentions))
        return "mention", mentions, rival_note(slugs)
    return "empty", [], miss_note(query, [])


def resolve_cars(conn, query: str, *, fuel=None, segment=None, seats_min=None,
                 price_max=None, limit=8):
    """rows, mention, candidates, none, or empty. A name is required for the last three."""
    status, payload, _note = classify(
        conn, query, fuel=fuel, segment=segment,
        seats_min=seats_min, price_max=price_max, limit=limit,
    )
    if status == "none":
        return "none", []
    if status == "candidates":
        return "candidates", payload
    return status, payload


def gate(conn, car: str | None):
    """Block a search, or limit it to one family's subject files. No car means no limit."""
    if not car or not compact(car):
        return None, None
    status, payload, note = classify(conn, car)
    if status == "rows":
        files = list(dict.fromkeys(row["file"] for row in payload))
        return None, files
    if note:
        return note, None
    return miss_note(car, []), None


def same_file(left: str, right: str) -> bool:
    def key(name: str) -> str:
        return name.removesuffix(".md")
    return key(left) == key(right)


def rebuild_aliases(conn) -> None:
    cars = conn.execute("SELECT * FROM cars").fetchall()
    conn.execute("DELETE FROM aliases")
    for key, family_id, kind in alias_records(cars):
        conn.execute(
            "INSERT OR IGNORE INTO aliases (key, family_id, kind) VALUES (?, ?, ?)",
            (key, family_id, kind),
        )
    labels = []
    seen = set()
    for car in cars:
        family_id = car["family_id"]
        if not family_id or family_id in seen:
            continue
        seen.add(family_id)
        labels.append((family_id, compact(f"{car['make']} {car['model']}")))
        if car["name_he"]:
            labels.append((family_id, compact(car["name_he"])))
    confirmed = confirmed_pairs()
    for left, right, distance in possible_duplicates(labels):
        if (left, right) in confirmed:
            continue
        log("possible_duplicate", left=left, right=right, distance=distance)
    conn.commit()


def confirmed_pairs(path: Path | None = None) -> set[tuple[str, str]]:
    path = path or (ROOT / "confirmed_distinct.json")
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {tuple(sorted(pair)) for pair in payload["pairs"]}
