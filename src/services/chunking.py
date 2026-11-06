"""Section-aware chunking of papers for embedding.

- Chunk 0 is always title + abstract, so every paper is searchable even without a parsed PDF.
- Full-text sections are packed into ~chunk_words chunks; small neighbouring sections share a
  chunk, long ones are split with overlap. Chunks never mix the end of one long section with
  the start of the next, so each chunk stays about one topic.
- References/acknowledgements are skipped: they match many queries and answer none.
"""

import re
from dataclasses import dataclass, field

from src.config import ChunkingSettings

SKIP_SECTIONS = re.compile(r"^\W*(references?|bibliography|acknowledge?ments?)\b", re.IGNORECASE)
ABSTRACT_SECTION = re.compile(r"^\W*abstract\b", re.IGNORECASE)


@dataclass
class Chunk:
    arxiv_id: str
    chunk_index: int
    section_titles: list[str]
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.arxiv_id}_{self.chunk_index}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class _Buffer:
    titles: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)


def embedding_text(title: str, chunk: Chunk) -> str:
    """What gets embedded: the chunk plus its paper title and section, so a chunk like
    'we fine-tune on 3 datasets' still carries what paper and topic it belongs to."""
    sections = "; ".join(chunk.section_titles)
    return f"{title}\nSection: {sections}\n\n{chunk.text}" if sections else f"{title}\n\n{chunk.text}"


def chunk_paper(
    arxiv_id: str,
    title: str,
    abstract: str,
    sections: list[dict] | None,
    settings: ChunkingSettings,
) -> list[Chunk]:
    if settings.overlap_words >= settings.chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    pieces: list[tuple[list[str], str]] = [(["Abstract"], f"{title}\n\n{abstract}".strip())]

    buffer = _Buffer()

    def flush() -> None:
        if buffer.words:
            pieces.append((buffer.titles, " ".join(buffer.words)))
        buffer.titles, buffer.words = [], []

    for section in sections or []:
        section_title = (section.get("title") or "").strip()
        words = (section.get("content") or "").split()
        if not words or SKIP_SECTIONS.match(section_title) or ABSTRACT_SECTION.match(section_title):
            continue
        if section_title == title.strip():  # Docling's first "section" is often just the title block
            continue

        if len(words) > settings.chunk_words:
            flush()
            step = settings.chunk_words - settings.overlap_words
            for start in range(0, len(words), step):
                window = words[start : start + settings.chunk_words]
                if start and len(window) <= settings.overlap_words:
                    break  # nothing new beyond the overlap
                pieces.append(([section_title], " ".join(window)))
            continue

        if len(buffer.words) + len(words) > settings.chunk_words:
            flush()
        buffer.titles.append(section_title)
        buffer.words.extend(words)
    flush()

    # Merge a tiny trailing piece into the one before it (never into the abstract chunk).
    if len(pieces) > 2 and len(pieces[-1][1].split()) < settings.min_words:
        titles, text = pieces.pop()
        prev_titles, prev_text = pieces[-1]
        pieces[-1] = (prev_titles + [t for t in titles if t not in prev_titles], f"{prev_text} {text}")

    return [Chunk(arxiv_id, i, titles, text) for i, (titles, text) in enumerate(pieces)]
