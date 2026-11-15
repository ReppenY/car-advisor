from car_advisor.eval_checks import attribute_problems, fact_problems


def test_explorer_price_must_not_be_the_calligraphy_price():
    review = "פורד אקספלורר עולה 365,000 שקלים ל-6 נוסעים ו-390,000 שקלים ל-7 נוסעים, כפי שרשום במבחן הפאליסייד."
    assert attribute_problems(review, [{
        "number": "390000",
        "owner": ["Explorer", "אקספלורר"],
        "not_owner": ["Calligraphy", "קליגרפי"],
        "required": False,
    }]) == []
    tested = "פאליסייד קליגרפי עולה 390,000 שקלים."
    problems = attribute_problems(tested, [{
        "number": "390000",
        "owner": ["Explorer", "אקספלורר"],
        "not_owner": ["Calligraphy", "קליגרפי"],
        "required": False,
    }])
    assert problems
    assert "קליגרפי" in problems[0]


def test_missing_optional_number_is_ok():
    text = "אקספלורר רשום ב-365,000 שקלים."
    assert attribute_problems(text, [{
        "number": "390000",
        "owner": ["אקספלורר"],
        "required": False,
    }]) == []


def test_suv_case_does_not_require_a_city_car_price():
    answer = "אין במבחנים רכב פנאי עד 150,000 שקלים. הזול ביותר הוא ג'ילי EX5 ב-162,000."
    assert fact_problems(answer, "geely-ex5.md", {"must_include": [{"number": "162000", "file": "geely-ex5.md"}]}) == []
