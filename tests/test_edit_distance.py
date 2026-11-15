from car_advisor.resolve import allowed_distance, compact, damerau_levenshtein


def test_known_edits():
    assert damerau_levenshtein("sportage", "sportige") == 1
    assert damerau_levenshtein("sportage", "sportgae") == 1
    assert damerau_levenshtein(compact("פאליסייד"), compact("פליסייד")) == 1


def test_allowed_distance_bands():
    assert allowed_distance(3) == 0
    assert allowed_distance(4) == 1
    assert allowed_distance(7) == 1
    assert allowed_distance(8) == 2
