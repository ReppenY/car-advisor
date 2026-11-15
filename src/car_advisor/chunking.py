import re

SIZE = 1200
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
PARA = re.compile(r"\n\s*\n")
SENTENCE = re.compile(r"(?<=[\.!?])\s+")


def _heading_before(text: str, offset: int) -> str:
    found = ""
    for match in HEADING.finditer(text):
        if match.start() > offset:
            break
        found = match.group(1)
    return found


def _hard_slices(text: str, start: int) -> list[tuple[int, str]]:
    pieces = []
    rest = text
    offset = start
    while len(rest) > SIZE:
        cut = rest.rfind(" ", 0, SIZE)
        if cut < SIZE // 2:
            cut = SIZE
        pieces.append((offset, rest[:cut].strip()))
        rest = rest[cut:].strip()
        offset = start + (len(text) - len(rest))
    if rest.strip():
        pieces.append((offset, rest.strip()))
    return [(off, piece) for off, piece in pieces if piece]


def _split_long(text: str, start: int) -> list[tuple[int, str]]:
    if len(text) <= SIZE:
        return [(start, text.strip())] if text.strip() else []
    parts = SENTENCE.split(text)
    if len(parts) == 1:
        return _hard_slices(text, start)
    out = []
    cursor = start
    buf = ""
    buf_at = start
    for part in parts:
        located = text.find(part, cursor - start)
        at = start + located if located >= 0 else cursor
        if buf and len(buf) + 1 + len(part) > SIZE:
            out.extend(_split_long(buf, buf_at) if len(buf) > SIZE else [(buf_at, buf)])
            buf, buf_at = part.strip(), at
        else:
            if not buf:
                buf_at = at
            buf = f"{buf} {part}".strip() if buf else part.strip()
        cursor = at + len(part)
    if buf:
        out.extend(_hard_slices(buf, buf_at) if len(buf) > SIZE else [(buf_at, buf)])
    return [(off, piece) for off, piece in out if piece]


def chunk_text(text: str, size: int = SIZE) -> list[dict]:
    """Heading, then paragraph, then sentence. Nothing longer than `size` is kept whole."""
    spans = []
    last = 0
    for match in PARA.finditer(text):
        spans.append((last, text[last:match.start()]))
        last = match.end()
    spans.append((last, text[last:]))

    packed: list[tuple[int, str]] = []
    buf, buf_at = "", 0
    for at, para in spans:
        piece = para.strip()
        if not piece:
            continue
        if buf and len(buf) + 2 + len(piece) > size:
            packed.append((buf_at, buf))
            buf, buf_at = piece, at
        else:
            if not buf:
                buf_at = at
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        packed.append((buf_at, buf))

    flat: list[tuple[int, str]] = []
    for at, piece in packed:
        flat.extend(_split_long(piece, at) if len(piece) > size else [(at, piece)])

    chunks = []
    for offset, body in flat:
        start_line = text[:offset].count("\n") + 1
        chunks.append({
            "text": body,
            "heading": _heading_before(text, offset),
            "start_line": start_line,
            "end_line": start_line + body.count("\n"),
        })
    return chunks
