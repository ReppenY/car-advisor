import tempfile
from pathlib import Path

from car_advisor.chunking import chunk_text
from car_advisor.ingest import index_article


def test_long_paragraph_is_split():
    text = "שורה. " * 400
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 1200 for chunk in chunks)
    assert chunks[0]["start_line"] >= 1


def test_unchanged_article_is_not_embedded():
    calls = {"n": 0}

    def fake_embed(texts, query=False):
        calls["n"] += 1
        return [[float(len(text)), 0.0, 0.0, 1.0] for text in texts]

    body = "source: https://example.test/car\n\n" + ("המחיר 189,000 שקלים. הנוחות טובה בנסיעה עירונית. " * 40)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert index_article("demo", "https://example.test/car", body, root, fake_embed) == "embedded"
        assert index_article("demo", "https://example.test/car", body, root, fake_embed) == "skipped"
    assert calls["n"] == 1


if __name__ == "__main__":
    test_long_paragraph_is_split()
    test_unchanged_article_is_not_embedded()
    print("corpus ok")
