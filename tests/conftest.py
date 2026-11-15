import uuid

from car_advisor.resolve import compact
from car_advisor.store import connect, rebuild_aliases


def seed(tmp_path):
    conn = connect(tmp_path)
    return conn


def add(conn, **row):
    make = row["make"]
    model = row["model"]
    item = {
        "id": uuid.uuid4().hex[:12],
        "slug": row.get("slug", "review"),
        "make": make,
        "model": model,
        "segment": row.get("segment", "suv"),
        "fuel": row.get("fuel", "petrol"),
        "seats": row.get("seats"),
        "price_nis": row.get("price_nis"),
        "source_line": row.get("source_line", 1),
        "file": row.get("file") or f"{row.get('slug', 'review')}.md",
        "role": row.get("role", "subject"),
        "family_id": row.get("family_id") or compact(f"{make} {model}"),
        "name_he": row.get("name_he", ""),
        "name_en": row.get("name_en") or f"{make} {model}",
        "trim_he": row.get("trim_he", ""),
        "trim_en": row.get("trim_en", ""),
        "engine": row.get("engine", ""),
    }
    conn.execute(
        """INSERT INTO cars (
            id, slug, make, model, segment, fuel, seats, price_nis, source_line, file, role,
            family_id, name_he, name_en, trim_he, trim_en, engine
        ) VALUES (
            :id, :slug, :make, :model, :segment, :fuel, :seats, :price_nis, :source_line, :file, :role,
            :family_id, :name_he, :name_en, :trim_he, :trim_en, :engine
        )""",
        item,
    )
    return item


def load(conn):
    rebuild_aliases(conn)
    return conn
