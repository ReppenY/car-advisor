import hashlib
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import trafilatura

from car_advisor.articles import ARTICLES
from car_advisor.chunking import chunk_text
from car_advisor.log import log
from car_advisor.numbers import find_mentions
from car_advisor.paths import ROOT
from car_advisor.retrieval import MODEL, collection, embed
from car_advisor.store import article_row, chunk_id, connect, content_hash, rebuild_aliases, replace_article

UA = "Mozilla/5.0 (compatible; car-advisor/0.1)"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def _title(markdown: str, slug: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    for line in markdown.splitlines():
        if line.strip() and not line.startswith("source:"):
            return line.strip()[:80]
    return slug


def save_article(slug: str, url: str, root: Path | None = None) -> str:
    """Download the HTML once and write the cleaned markdown. Returns the markdown."""
    root = root or ROOT
    raw_dir = root / "raw"
    article_dir = root / "articles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    article_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{slug}.html"
    if raw_path.exists():
        html = raw_path.read_bytes()
    else:
        html = _download(url)
        raw_path.write_bytes(html)
        log("download", slug=slug, bytes=len(html))
    markdown = trafilatura.extract(
        html, output_format="markdown", include_comments=False, favor_precision=True, url=url,
    )
    if not markdown or len(markdown) < 500:
        raise RuntimeError(f"{slug}: cleaned article is too short ({len(markdown or '')} chars)")
    document = f"source: {url}\n\n{markdown.strip()}\n"
    (article_dir / f"{slug}.md").write_text(document, encoding="utf-8")
    return document


def index_article(slug: str, url: str, document: str, root: Path | None = None,
                  embed_fn=embed) -> str:
    """Embed and store one article. Returns 'skipped' or 'embedded'."""
    root = root or ROOT
    digest = content_hash(document)
    conn = connect(root)
    try:
        row = article_row(conn, slug)
        if row and row["content_hash"] == digest and row["embedded_model"] == MODEL:
            log("ingest_skip", slug=slug)
            return "skipped"
        title = _title(document, slug)
        chunks = []
        for piece in chunk_text(document):
            chunks.append({
                **piece,
                "id": chunk_id(slug, piece["text"]),
                "mentions": find_mentions(piece["text"]),
            })
        vectors = embed_fn([f"{title}\n{chunk['heading']}\n{chunk['text']}".strip() for chunk in chunks])
        col = collection(root)
        existing = col.get(where={"slug": slug}, include=[])
        if existing["ids"]:
            col.delete(ids=existing["ids"])
        col.upsert(
            ids=[chunk["id"] for chunk in chunks],
            embeddings=vectors,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[{
                "slug": slug,
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "heading": chunk["heading"] or "",
            } for chunk in chunks],
        )
        replace_article(conn, slug, url, title, digest, MODEL, chunks)
    finally:
        conn.close()
    log("ingest", slug=slug, chunks=len(chunks))
    return "embedded"


def _slug_for(conn, url: str) -> str:
    leaf = urlparse(url).path.strip("/").split("/")[-1]
    base = re.sub(r"[^a-z0-9]+", "-", leaf.lower()).strip("-")[:60]
    if len(base) < 2:
        raise ValueError("That URL has no article name.")
    row = article_row(conn, base)
    if row is None or row["url"] == url:
        return base
    suffix = hashlib.sha256(url.encode()).hexdigest()[:6]
    return f"{base}-{suffix}"


def add_url(url: str) -> dict:
    """Store one new article. A URL already in the catalog is left as it is."""
    from car_advisor.extract import extract_article, save_extracted

    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Give a full http or https article URL.")
    conn = connect()
    try:
        existing = conn.execute(
            "SELECT slug, title FROM articles WHERE url = ?", (url,)
        ).fetchone()
        slug = None if existing else _slug_for(conn, url)
    finally:
        conn.close()
    if existing:
        return {"slug": existing["slug"], "title": existing["title"], "status": "already stored"}
    document = save_article(slug, url)
    status = index_article(slug, url, document)
    rows = extract_article(slug, document)
    conn = connect()
    try:
        save_extracted(conn, slug, rows)
        rebuild_aliases(conn)
        title = article_row(conn, slug)["title"]
    finally:
        conn.close()
    return {"slug": slug, "title": title, "status": status}


def ingest_all(root: Path | None = None, embed_fn=embed) -> list[str]:
    results = []
    for slug, url in ARTICLES:
        document = save_article(slug, url, root)
        results.append(index_article(slug, url, document, root, embed_fn))
    conn = connect(root)
    try:
        rebuild_aliases(conn)
    finally:
        conn.close()
    return results


if __name__ == "__main__":
    ingest_all()
