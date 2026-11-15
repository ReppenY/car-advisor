"""Quality of extraction, names, and retrieval on two articles that were not in the tuned set.

Uses the existing functions. Does not edit them.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_advisor.extract import extract_article  # noqa: E402
from car_advisor.ingest import index_article  # noqa: E402
from car_advisor.resolve import alias_records, compact, match_aliases  # noqa: E402
from car_advisor.retrieval import search  # noqa: E402
from car_advisor.store import connect, rebuild_aliases  # noqa: E402

HOLD = ROOT / "data" / "holdout"
GOLD = ROOT / "eval" / "holdout_gold.json"
OUT = ROOT / "eval" / "holdout_metrics.json"


def family(row) -> str:
    return row["family_id"] or compact(f"{row['make']} {row['model']}")


def role_scores(rows, gold_articles):
    tp = fp = fn = 0
    mismatches = []
    for article in gold_articles:
        got = {family(row) for row in rows if row["slug"] == article["slug"] and row["role"] == "subject"}
        gold = {article["subject"]}
        for name in sorted(got & gold):
            tp += 1
        for name in sorted(got - gold):
            fp += 1
            mismatches.append({"file": f"{article['slug']}.md", "kind": "false_subject", "family": name})
        for name in sorted(gold - got):
            fn += 1
            mismatches.append({"file": f"{article['slug']}.md", "kind": "missed_subject", "family": name})
        mentioned = set(article["mentioned"])
        for row in rows:
            if row["slug"] != article["slug"] or row["role"] != "subject":
                continue
            if family(row) in mentioned:
                mismatches.append({"file": f"{article['slug']}.md", "kind": "rival_marked_subject", "family": family(row)})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "mismatches": mismatches}


def price_recall(rows, gold_articles):
    found = missed = 0
    missing = []
    for article in gold_articles:
        have = {
            row["price_nis"]
            for row in rows
            if row["slug"] == article["slug"] and row["role"] == "subject"
        }
        for price in article["prices"]:
            if price in have:
                found += 1
            else:
                missed += 1
                missing.append({"file": f"{article['slug']}.md", "price": price})
    return {
        "recall": round(found / (found + missed), 4) if found + missed else 0.0,
        "missing": missing,
    }


def resolver_scores(rows, names):
    aliases = alias_records(rows)
    tp = fp = fn = 0
    wrong = []
    for item in names:
        found = match_aliases(item["name"], aliases)
        predicted = found.family_ids[0] if found.status == "one" else None
        gold = item["expect"]
        if found.status == "one" and predicted == gold:
            tp += 1
        elif found.status == "one":
            fp += 1
            fn += 1
            wrong.append({**item, "predicted": predicted})
        else:
            fn += 1
            wrong.append({**item, "predicted": found.status})
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "wrong": wrong,
    }


def store_rows(rows):
    conn = connect(HOLD)
    try:
        conn.execute("DELETE FROM cars")
        for row in rows:
            conn.execute(
                """INSERT INTO cars (
                    id, slug, make, model, segment, fuel, seats, price_nis, source_line, file, role,
                    family_id, name_he, name_en, trim_he, trim_en, engine
                ) VALUES (
                    :id, :slug, :make, :model, :segment, :fuel, :seats, :price_nis, :source_line, :file, :role,
                    :family_id, :name_he, :name_en, :trim_he, :trim_en, :engine
                )""",
                row,
            )
        conn.commit()
        rebuild_aliases(conn)
    finally:
        conn.close()


def retrieval_scores(queries):
    rows = []
    for item in queries:
        hits = search(item["query"], k=4, root=HOLD)
        prices = ("199,000", "199000", "220,000", "220000") if "ev5" in item["id"] else ("185,000", "185000", "200,000", "200000")
        relevant = [
            hit for hit in hits
            if hit["file"] == item["file"] and any(price in hit["text"].replace(",", "") or price in hit["text"] for price in prices)
        ]
        # A returned passage is relevant when it is from the asked file and states a gold price.
        labeled = []
        for rank, hit in enumerate(hits, start=1):
            text = hit["text"]
            has_price = any(price in text or price in text.replace(",", "") for price in prices)
            ok = hit["file"] == item["file"] and has_price
            labeled.append({
                "rank": rank,
                "file": hit["file"],
                "lines": f"{hit['start_line']}-{hit['end_line']}",
                "relevant": ok,
            })
        hit_at_1 = 1.0 if labeled and labeled[0]["relevant"] else 0.0
        hit_at_k = 1.0 if any(label["relevant"] for label in labeled) else 0.0
        precision = sum(label["relevant"] for label in labeled) / 4
        rows.append({
            "id": item["id"],
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
            "precision_at_4": round(precision, 4),
            "passages": labeled,
        })
    return rows


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    rows = []
    for article in gold["articles"]:
        text = (HOLD / "articles" / f"{article['slug']}.md").read_text(encoding="utf-8")
        extracted = extract_article(article["slug"], text)
        for row in extracted:
            row["slug"] = article["slug"]
        rows.extend(extracted)
        print(article["slug"], [(row["role"], row["name_en"], row["trim_en"], row["price_nis"], row["seats"]) for row in extracted], flush=True)
    store_rows(rows)
    for article in gold["articles"]:
        text = (HOLD / "articles" / f"{article['slug']}.md").read_text(encoding="utf-8")
        index_article(article["slug"], article["url"], text, HOLD)
    payload = {
        "roles": role_scores(rows, gold["articles"]),
        "prices": price_recall(rows, gold["articles"]),
        "resolver": resolver_scores(rows, gold["names"]),
        "retrieval": retrieval_scores(gold["queries"]),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "passages"} if isinstance(value, dict) else value for key, value in payload.items()}, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
