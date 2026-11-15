"""The saved subject list, checked against rows extraction wrote."""

from car_advisor.resolve import compact
from car_advisor.store import connect
from car_advisor.subjects import SUBJECTS


def _families(rows, role: str) -> dict[str, set[str]]:
    found = {}
    for row in rows:
        if row["role"] != role:
            continue
        found.setdefault(row["slug"], set()).add(compact(f"{row['make']} {row['model']}"))
    return found


def test_extracted_subjects_match_the_list():
    conn = connect()
    try:
        rows = conn.execute("SELECT slug, make, model, role, name_he, name_en FROM cars").fetchall()
    finally:
        conn.close()
    assert rows, "catalog has no cars; run python -m car_advisor.extract"
    assert all(row["role"] for row in rows), "catalog rows have no role; run python -m car_advisor.extract"
    got = _families(rows, "subject")
    expected = {slug: {compact(name) for name in names} for slug, names in SUBJECTS.items()}
    assert got == expected, f"subjects {got}"
    mentions = _families(rows, "mentioned")
    assert compact("Ford Explorer") in mentions.get("palisade", set())
    assert compact("Ford Explorer") not in got.get("palisade", set())
    subjects = {name for names in got.values() for name in names}
    assert compact("Tesla Model Y") not in subjects
    geometry = [
        row for row in rows
        if "geometry" in (row["model"] or "").casefold()
        or "גאומטרי" in (row["name_he"] or "")
        or "גאומטרי" in (row["model"] or "")
    ]
    assert geometry, "Geometry C was not extracted"
    assert all(row["role"] == "mentioned" for row in geometry)


if __name__ == "__main__":
    test_extracted_subjects_match_the_list()
    print("extract ok")
