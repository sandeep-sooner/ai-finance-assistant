"""Load the knowledge base from src/data and split it into retrievable chunks.

Articles live in src/data/knowledge_base/<category>.md. Each article is:

    # Title
    source: Publisher
    url: https://...

    Body paragraphs...

The glossary (src/data/glossary.json) is indexed too, one chunk per term, so
"what does beta mean?" retrieves the definition as well as the long article.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KB_DIR = DATA_DIR / "knowledge_base"
GLOSSARY_PATH = DATA_DIR / "glossary.json"

_META = re.compile(r"^(source|url):\s*(.+)$", re.I)
_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    source: str
    url: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str           # "<article id>#<n>"
    article_id: str
    title: str
    category: str
    source: str
    url: str | None
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def slugify(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")


def parse_articles(markdown: str, category: str) -> list[Article]:
    articles: list[Article] = []
    for block in re.split(r"^# ", markdown, flags=re.M)[1:]:
        lines = block.strip().splitlines()
        title, meta, body = lines[0].strip(), {}, []
        for line in lines[1:]:
            m = _META.match(line.strip()) if not body else None
            if m:
                meta[m.group(1).lower()] = m.group(2).strip()
            elif line.strip() or body:
                body.append(line)
        articles.append(Article(
            id=f"{category}/{slugify(title)}", title=title, category=category,
            source=meta.get("source", "Knowledge base"), url=meta.get("url"), text="\n".join(body).strip(),
        ))
    return articles


def load_articles(kb_dir: Path = KB_DIR) -> list[Article]:
    articles: list[Article] = []
    for path in sorted(kb_dir.glob("*.md")):
        articles.extend(parse_articles(path.read_text(encoding="utf-8"), path.stem))
    return articles


def load_glossary(path: Path = GLOSSARY_PATH) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return []


def split_text(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
    """Paragraph-aware chunking: pack whole paragraphs up to max_chars, and start each new chunk with
    the tail of the previous one so a fact that straddles a boundary is still retrievable."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > max_chars:  # a single oversized paragraph: cut at a sentence end
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            head, para = para[:cut].strip(), para[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(articles: list[Article] | None = None, glossary: list[dict] | None = None,
                 max_chars: int = 600, overlap: int = 100) -> list[Chunk]:
    articles = load_articles() if articles is None else articles
    glossary = load_glossary() if glossary is None else glossary
    chunks: list[Chunk] = []
    for a in articles:
        for i, piece in enumerate(split_text(a.text, max_chars, overlap)):
            chunks.append(Chunk(f"{a.id}#{i}", a.id, a.title, a.category, a.source, a.url, piece))
    for g in glossary:
        gid = f"glossary/{slugify(g['term'])}"
        chunks.append(Chunk(
            f"{gid}#0", gid, f"Glossary: {g['term']}", g.get("category", "glossary"), "Glossary", None,
            f"{g['term']}: {g['definition']}",
        ))
    return chunks


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Changes whenever any chunk changes, so a stale on-disk index is rebuilt automatically."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]
