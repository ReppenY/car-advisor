import io
from contextlib import redirect_stderr

from car_advisor.resolve import compact, possible_duplicates
from car_advisor.store import confirmed_pairs, rebuild_aliases
from tests.conftest import add, seed


def test_near_duplicates_are_flagged_and_not_merged():
    labels = [
        ("xpengg6", compact("Xpeng G6")),
        ("xpengg9", compact("Xpeng G9")),
        ("hyundaipalisade", compact("Hyundai Palisade")),
        ("hyundiapalisade", compact("Hyundia Palisade")),
    ]
    pairs = {(left, right): distance for left, right, distance in possible_duplicates(labels)}
    assert pairs[("xpengg6", "xpengg9")] == 1
    assert ("hyundaipalisade", "hyundiapalisade") in pairs
    assert {"xpengg6", "xpengg9", "hyundaipalisade", "hyundiapalisade"} == {item[0] for item in labels}


def test_confirmed_pairs_are_not_flagged_again(tmp_path):
    confirmed = confirmed_pairs()
    assert ("xpengg6", "xpengg9") in confirmed
    assert ("xpengg6", "xpengp7") in confirmed
    assert ("xpengg9", "xpengp7") in confirmed
    assert ("skodakamiq", "skodakaroq") in confirmed
    assert ("volkswagengolf", "volkswagenpolo") in confirmed
    conn = seed(tmp_path)
    add(conn, make="Xpeng", model="G6", name_he="אקספנג G6")
    add(conn, make="Xpeng", model="G9", name_he="אקספנג G9")
    add(conn, make="Hyundai", model="Palisade", name_he="יונדאי פאליסייד")
    add(conn, make="Hyundia", model="Palisade", name_he="היונדיה פאליסייד")
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        rebuild_aliases(conn)
    text = buffer.getvalue()
    assert "xpengg6" not in text
    assert "hyundiapalisade" in text
    assert '"event": "possible_duplicate"' in text
    stored = {row["family_id"] for row in conn.execute("SELECT family_id FROM cars")}
    assert {"xpengg6", "xpengg9", "hyundaipalisade", "hyundiapalisade"} <= stored
