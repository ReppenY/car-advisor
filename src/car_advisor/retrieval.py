import re
from pathlib import Path

from chromadb import PersistentClient
from rank_bm25 import BM25Okapi

from car_advisor.paths import ROOT
from car_advisor.resolve import HEBREW_PREFIXES
from car_advisor.store import all_chunks, connect

MODEL = "intfloat/multilingual-e5-large"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
TOKEN = re.compile(r"[א-ת0-9]+|[A-Za-z0-9]+")
RRF_K = 60

_embedder = None
_bm25_key = None
_bm25 = None
_bm25_rows = None


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN.findall(text):
        token = raw.lower()
        tokens.append(token)
        if len(token) >= 4 and token[0] in HEBREW_PREFIXES and all("א" <= char <= "ת" for char in token):
            tokens.append(token[1:])
    return tokens


def embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(MODEL)
    return _embedder


def embed(texts: list[str], *, query: bool = False) -> list[list[float]]:
    prefix = QUERY_PREFIX if query else PASSAGE_PREFIX
    vectors = embedder().embed([prefix + text for text in texts])
    return [vector.tolist() for vector in vectors]


def collection(root: Path | None = None):
    root = root or ROOT
    path = root / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(path))
    names = [item.name for item in client.list_collections()]
    if "passages" in names:
        current = client.get_collection("passages")
        if (current.metadata or {}).get("model") == MODEL:
            return current
        client.delete_collection("passages")
    return client.create_collection("passages", metadata={"hnsw:space": "cosine", "model": MODEL})


def _index(conn, root: Path):
    global _bm25_key, _bm25, _bm25_rows
    rows = all_chunks(conn)
    key = (str(root), tuple(row["id"] for row in rows))
    if key == _bm25_key:
        return _bm25, _bm25_rows
    tokenized = [tokenize(row["text"]) or ["_"] for row in rows]
    _bm25_key = key
    _bm25_rows = rows
    _bm25 = BM25Okapi(tokenized) if rows else None
    return _bm25, rows


def _rrf(rankings: list[list[str]]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [chunk_id for chunk_id, _ in sorted(scores.items(), key=lambda item: -item[1])]


def _diversify(ordered: list[str], by_id: dict, k: int) -> list[str]:
    buckets: dict[str, list[str]] = {}
    slugs: list[str] = []
    for chunk_id in ordered:
        slug = by_id[chunk_id]["slug"]
        if slug not in buckets:
            slugs.append(slug)
            buckets[slug] = []
        buckets[slug].append(chunk_id)
    picked = []
    while len(picked) < k and any(buckets.values()):
        for slug in slugs:
            if buckets[slug]:
                picked.append(buckets[slug].pop(0))
            if len(picked) >= k:
                break
    return picked


def search(query: str, *, k: int = 4, files: list[str] | None = None, root: Path | None = None) -> list[dict]:
    root = root or ROOT
    conn = connect(root)
    try:
        index, rows = _index(conn, root)
    finally:
        conn.close()
    if not rows:
        return []
    by_id = {row["id"]: row for row in rows}
    slugs = {name.removesuffix(".md") for name in files} if files else None
    pool = max(k * 5, 20)

    dense_ids: list[str] = []
    col = collection(root)
    count = col.count()
    if count:
        where = None
        if slugs:
            wanted = list(slugs)
            where = {"slug": wanted[0]} if len(wanted) == 1 else {"slug": {"$in": wanted}}
        found = col.query(
            query_embeddings=embed([query], query=True),
            n_results=min(pool, count),
            where=where,
        )
        dense_ids = [chunk_id for chunk_id in found["ids"][0] if chunk_id in by_id]

    lexical_ids: list[str] = []
    if index is not None:
        scores = index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        for position in ranked:
            if scores[position] <= 0:
                break
            row = rows[position]
            if slugs and row["slug"] not in slugs:
                continue
            lexical_ids.append(row["id"])
            if len(lexical_ids) >= pool:
                break

    ordered = _diversify(_rrf([dense_ids, lexical_ids]), by_id, k)
    hits = []
    for chunk_id in ordered:
        row = by_id[chunk_id]
        hits.append({
            "file": f"{row['slug']}.md",
            "slug": row["slug"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "heading": row["heading"],
            "text": row["text"],
        })
    return hits


def format_hit(hit: dict) -> str:
    lines = f"{hit['start_line']}-{hit['end_line']}"
    heading = f" ({hit['heading']})" if hit["heading"] else ""
    return f"[source: {hit['file']} lines {lines}]{heading}\n{hit['text']}"
