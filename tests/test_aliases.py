from car_advisor.store import connect
from tests.conftest import add, load, seed


def families_for(conn, key):
    return {row["family_id"] for row in conn.execute("SELECT family_id FROM aliases WHERE key = ?", (key,))}


def test_model_alias_is_unique_until_a_second_family_is_added(tmp_path):
    conn = seed(tmp_path)
    add(conn, make="Kia", model="Sportage", name_he="קיה ספורטאז'")
    load(conn)
    assert families_for(conn, "sportage") == {"kiasportage"}
    assert families_for(conn, "kia") == {"kiasportage"}
    assert families_for(conn, "קיה") == {"kiasportage"}
    add(conn, make="Other", model="Sportage", name_he="אחר ספורטאז'")
    load(conn)
    assert families_for(conn, "sportage") == {"kiasportage", "othersportage"}


def test_trim_and_article_spelling_point_at_the_family(tmp_path):
    conn = seed(tmp_path)
    add(conn, make="Hyundai", model="Palisade", name_he="יונדאי פאליסייד", trim_en="Calligraphy", trim_he="קליגרפי")
    load(conn)
    assert families_for(conn, "calligraphy") == {"hyundaipalisade"}
    assert families_for(conn, "קליגרפי") == {"hyundaipalisade"}
    assert families_for(conn, "יונדאיפאליסייד") == {"hyundaipalisade"}
    assert families_for(conn, "פאליסייד") == {"hyundaipalisade"}


def test_rebuild_uses_the_connection_catalog(tmp_path):
    first = seed(tmp_path / "one")
    add(first, make="Kia", model="Sportage")
    load(first)
    second = connect(tmp_path / "two")
    add(second, make="Mazda", model="2", name_he="מאזדה 2")
    load(second)
    assert families_for(first, "sportage") == {"kiasportage"}
    assert families_for(second, "mazda2") == {"mazda2"}
    assert families_for(second, "sportage") == set()
