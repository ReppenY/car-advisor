from car_advisor.numbers import covers, find_mentions


def forms(text):
    mentions = find_mentions(text)
    assert len(mentions) == 1, mentions
    return mentions[0]


def test_alef_permutations():
    mention = forms("189אלף")
    assert mention.canonical == "189000"
    for form in ("189אלף", "189k", "189000", "189,000", "189 אלף", "₪189,000"):
        assert form in mention.forms, form


def test_grouped_price_matches_alef_spelling():
    article = forms("189,900")
    answer = forms("189.9 אלף")
    assert article.canonical == answer.canonical == "189900"
    assert not article.forms.isdisjoint(answer.forms)


def test_nearby_thousand_does_not_match():
    article = forms("189,900")
    answer = forms("190 אלף")
    assert article.forms.isdisjoint(answer.forms)


def test_small_number_stays_small():
    mention = forms("7")
    assert mention.canonical == "7"
    assert "7000" not in mention.forms
    assert "7 אלף" not in mention.forms


def test_decimal_consumption():
    mention = forms("14.5")
    assert mention.canonical == "14.5"
    assert mention.forms == frozenset({"14.5"})


def test_shekels_plural():
    mention = forms("255,000 שקלים")
    assert mention.canonical == "255000"
    assert "255 אלף" in mention.forms
    assert "255,000 שקלים" in mention.forms


def test_hebrew_prefix_keeps_a_price():
    assert any(mention.canonical == "411000" for mention in find_mentions("ב-411,000 שקלים"))
    assert any(mention.canonical == "411000" for mention in find_mentions("ו-411 אלף שקל"))


def test_hebrew_hyphen_keeps_a_seat_count():
    assert any(mention.canonical == "6" for mention in find_mentions("ל-6 נוסעים"))
    assert find_mentions("Mazda CX-5") == []


def test_model_names_and_list_markers_are_not_claims():
    assert find_mentions("Xpeng G9, Xpeng G6, Geely EX5, Mazda CX-5") == []
    assert find_mentions("1. Stretch the budget\n2. Look at family cars") == []
    assert forms("6.95").canonical == "6.95"


def test_session_rejection_keeps_only_the_real_price():
    """The logged rejection was '9, 1, 160,000, 2'. Only 160,000 is a real miss."""
    allowed = set()
    for text in ("162,000 שקלים", "168,000 שקלים", "about 150k NIS"):
        for mention in find_mentions(text):
            allowed |= mention.forms
    answer = (
        "Nothing at or below 150,000 NIS.\n"
        "Geely EX5 Pro - 162,000 NIS. Geely EX5 Tech - 168,000 NIS.\n"
        "Other SUVs (Xpeng G9, Tesla Model Y) are listed without a price.\n"
        "1. Stretching your budget slightly to around 160,000 NIS\n"
        "2. Looking at family cars"
    )
    assert covers(answer, allowed) == ["160,000"]


def test_answer_check_allows_user_numbers():
    allowed = set()
    for mention in find_mentions("עד 200 אלף"):
        allowed |= mention.forms
    assert covers("התקציב שלך הוא 200,000", allowed) == []
    assert covers("המחיר 168,000", allowed) == ["168,000"]


if __name__ == "__main__":
    test_alef_permutations()
    test_grouped_price_matches_alef_spelling()
    test_nearby_thousand_does_not_match()
    test_small_number_stays_small()
    test_decimal_consumption()
    test_shekels_plural()
    test_hebrew_prefix_keeps_a_price()
    test_hebrew_hyphen_keeps_a_seat_count()
    test_model_names_and_list_markers_are_not_claims()
    test_session_rejection_keeps_only_the_real_price()
    test_answer_check_allows_user_numbers()
    print("numbers ok")
