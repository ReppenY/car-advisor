"""Extracted roles against the saved subject list. Uses the catalog on disk."""

from car_advisor.paths import ARTICLES
from car_advisor.resolve import compact, name_in_title
from car_advisor.store import connect
from car_advisor.subjects import SUBJECTS


def rows():
    conn = connect()
    try:
        return conn.execute("SELECT slug, make, model, role, name_he, name_en FROM cars").fetchall()
    finally:
        conn.close()


def test_subjects_match_the_ground_truth():
    found = rows()
    assert found, "catalog has no cars; run python -m car_advisor.extract"
    got = {}
    for row in found:
        if row["role"] != "subject":
            continue
        got.setdefault(row["slug"], set()).add(compact(f"{row['make']} {row['model']}"))
    expected = {slug: {compact(name) for name in names} for slug, names in SUBJECTS.items()}
    assert got == expected
    mentions = {}
    for row in found:
        if row["role"] == "mentioned":
            mentions.setdefault(row["slug"], set()).add(compact(f"{row['make']} {row['model']}"))
    assert compact("Ford Explorer") in mentions["palisade"]
    assert compact("Chevrolet Traverse") in mentions["palisade"]
    assert compact("Ford Explorer") not in got["palisade"]


def test_subject_role_precision_and_recall():
    from role_metrics import role_metrics

    metrics = role_metrics()
    assert metrics["mismatches"] == []
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1


def test_subject_name_appears_in_the_title():
    for row in rows():
        if row["role"] != "subject":
            continue
        article = (ARTICLES / f"{row['slug']}.md").read_text(encoding="utf-8")
        assert name_in_title(row["name_he"], row["name_en"], article), row["name_en"]
