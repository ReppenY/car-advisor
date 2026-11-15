from car_advisor.resolve import compact


def test_mazda_spellings_share_a_key():
    assert compact("Mazda2") == compact("Mazda 2") == compact("mazda-2") == "mazda2"


def test_hebrew_compact_key():
    assert compact("פּוֹלוֹ") == compact("פולו")
    assert compact("ספורטאז'") == compact("ספורטאז")
    assert compact("ספורטאז׳") == compact('ספורטאז"')
    assert compact("ספורטאז״") == compact("ספורטאז")
    assert compact("סוף") == compact("סופ")
    assert compact("מלך") == compact("מלכ")
    assert compact("אם") == compact("אמ")
    assert compact("כן") == compact("כנ")
    assert compact("אף") == compact("אפ")
    assert compact("עץ") == compact("עצ")
