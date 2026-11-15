"""Resolver on a fixture catalog. No model calls."""

import io
from contextlib import redirect_stderr

from car_advisor.graph import scan_subject
from car_advisor.store import classify, resolve_cars
from tests.conftest import add, load, seed


def catalog(tmp_path):
    conn = seed(tmp_path)
    palisade = dict(slug="palisade", make="Hyundai", model="Palisade", name_he="יונדאי פאליסייד", file="palisade.md")
    add(conn, **palisade, trim_en="Excellence", trim_he="אקסלנס", price_nis=350000)
    add(conn, **palisade, trim_en="Excellence", trim_he="אקסלנס", price_nis=358000)
    add(conn, **palisade, trim_en="Calligraphy", trim_he="קליגרפי", price_nis=390000)
    add(conn, **palisade, trim_en="Calligraphy", trim_he="קליגרפי", price_nis=411000)
    add(conn, slug="palisade", make="Ford", model="Explorer", name_he="פורד אקספלורר", role="mentioned", price_nis=365000, file="palisade.md")
    add(conn, slug="yaris-mazda2", make="Mazda", model="2", name_he="מאזדה 2", trim_en="Dynamic", price_nis=126000, file="yaris-mazda2.md")
    add(conn, slug="xpeng-g6", make="Xpeng", model="G6", name_he="אקספנג G6", price_nis=255000, file="xpeng-g6.md")
    add(conn, slug="tucson-sportage", make="Kia", model="Sportage", name_he="קיה ספורטאז'", price_nis=210000, file="tucson-sportage.md")
    add(conn, slug="polo", make="Volkswagen", model="Polo", name_he="פולקסווגן פולו", trim_en="Style", trim_he="סטייל", price_nis=129000, file="polo.md")
    add(conn, slug="yaris-mazda2", make="Toyota", model="Yaris", name_he="טויוטה יאריס", trim_en="Style", trim_he="סטייל", price_nis=142000, file="yaris-mazda2.md")
    return load(conn)


def prices(rows):
    return {row["price_nis"] for row in rows}


def test_exact_and_cross_script(tmp_path):
    conn = catalog(tmp_path)
    assert prices(resolve_cars(conn, "Hyundai Palisade")[1]) == {350000, 358000, 390000, 411000}
    assert prices(resolve_cars(conn, "palisade")[1]) == {350000, 358000, 390000, 411000}
    assert prices(resolve_cars(conn, "פאליסייד")[1]) == {350000, 358000, 390000, 411000}
    assert prices(resolve_cars(conn, "mazda2")[1]) == {126000}
    assert prices(resolve_cars(conn, "מאזדה 2")[1]) == {126000}
    assert prices(resolve_cars(conn, "אקספנג G6")[1]) == {255000}
    assert prices(resolve_cars(conn, "Xpeng G6")[1]) == {255000}


def test_typos_within_the_allowed_distance(tmp_path):
    conn = catalog(tmp_path)
    assert resolve_cars(conn, "sportige")[1][0]["model"] == "Sportage"
    assert resolve_cars(conn, "sportgae")[1][0]["model"] == "Sportage"


def test_short_names_are_exact_only(tmp_path):
    conn = catalog(tmp_path)
    assert prices(resolve_cars(conn, "g6")[1]) == {255000}
    status, rows = resolve_cars(conn, "g7")
    assert status == "none"
    assert rows == []


def test_trim_returns_that_trims_prices(tmp_path):
    # Excellence is 350,000 and 358,000. Calligraphy is 390,000 and 411,000.
    conn = catalog(tmp_path)
    assert prices(resolve_cars(conn, "Calligraphy")[1]) == {390000, 411000}
    assert prices(resolve_cars(conn, "קליגרפי")[1]) == {390000, 411000}
    assert prices(resolve_cars(conn, "Excellence")[1]) == {350000, 358000}


def test_shared_alias_returns_both_families(tmp_path):
    conn = catalog(tmp_path)
    status, families = resolve_cars(conn, "Style")
    assert status == "candidates"
    assert set(families) == {"volkswagenpolo", "toyotayaris"}


def test_two_families_inside_the_distance_are_candidates(tmp_path):
    conn = seed(tmp_path)
    add(conn, make="Kia", model="Sportage", name_he="קיה ספורטאז'")
    add(conn, make="Other", model="Sportige", name_he="אחר")
    load(conn)
    status, families = resolve_cars(conn, "sportxge")
    assert status == "candidates"
    assert set(families) == {"kiasportage", "othersportige"}


def test_mention_only(tmp_path):
    conn = catalog(tmp_path)
    status, rows = resolve_cars(conn, "Explorer")
    assert status == "mention"
    assert {row["slug"] for row in rows} == {"palisade"}
    from car_advisor.store import rival_note
    note = rival_note(["palisade"])
    assert note.startswith("Not reviewed. Mentioned in: palisade (as a rival).")


def test_unknown_is_not_covered_and_logged(tmp_path):
    # This fixture has no Model Y row. The extracted catalog mentions it; see test_tools.
    conn = catalog(tmp_path)
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        status, rows = resolve_cars(conn, "Model Y")
    assert status == "none"
    assert rows == []
    assert '"event": "unresolved"' in buffer.getvalue()
    assert "Model Y" in buffer.getvalue()


def test_polo_shirt_does_not_resolve(tmp_path):
    conn = catalog(tmp_path)
    assert resolve_cars(conn, "polo shirt")[0] == "none"


def makes(tmp_path):
    conn = seed(tmp_path)
    add(conn, make="Peugeot", model="408", name_he="פיג'ו 408", name_en="Peugeot 408",
        price_nis=170000, file="peugeot-408-fl.md", slug="peugeot-408-fl")
    add(conn, make="Peugeot", model="402", name_he="פיג'ו 402", role="mentioned",
        file="peugeot-408-fl.md", slug="peugeot-408-fl")
    add(conn, make="Hyundai", model="Palisade", name_he="יונדאי פאליסייד", file="palisade.md", slug="palisade")
    add(conn, make="Hyundai", model="Tucson", name_he="יונדאי טוסון", file="tucson-sportage.md", slug="tucson-sportage")
    add(conn, make="Mazda", model="2", name_he="מאזדה 2", file="yaris-mazda2.md", slug="yaris-mazda2")
    return load(conn)


def test_a_make_with_one_reviewed_model_answers_about_it(tmp_path):
    conn = makes(tmp_path)
    for name in ("Peugeot", "פיג'ו", "פג'ו", "פיגו"):
        status, rows = resolve_cars(conn, name)
        assert status == "rows", name
        assert {row["model"] for row in rows} == {"408"}
        assert all(row["role"] == "subject" for row in rows)
    status, rows, note = classify(conn, "פג'ו")
    assert status == "rows"
    assert note == "We have the Peugeot 408 review."
    found = scan_subject(conn, "כמה עולה פיג'ו?")
    assert found["family_id"] == "peugeot408"
    assert scan_subject(conn, "מה המחיר של מאזדה CX-5?") is None


def test_a_make_with_two_reviewed_models_asks_which(tmp_path):
    conn = makes(tmp_path)
    for name in ("Hyundai", "יונדאי"):
        status, families = resolve_cars(conn, name)
        assert status == "candidates", name
        assert set(families) == {"hyundaipalisade", "hyundaitucson"}
    _status, _families, note = classify(conn, "Hyundai")
    assert note.startswith("Which reviewed model do you mean?")
    assert "Palisade" in note and "Tucson" in note
    assert "402" not in note
    assert scan_subject(conn, "מה יש מיונדאי?") is None


def test_a_miss_names_close_cars_and_does_not_return_them(tmp_path):
    conn = catalog(tmp_path)
    status, rows, note = classify(conn, "g7")
    assert status == "none"
    assert rows == []
    assert note.startswith('I couldn\'t find "g7" by that name.')
    assert "not a match" in note
    assert "G6" in note
    status, rows = resolve_cars(conn, "Mazda CX-5")
    assert status == "none"
    assert rows == []


def test_near_miss_is_logged(tmp_path):
    conn = catalog(tmp_path)
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        resolve_cars(conn, "g7")
    text = buffer.getvalue()
    assert '"event": "near_miss"' in text
    assert '"key": "g6"' in text
