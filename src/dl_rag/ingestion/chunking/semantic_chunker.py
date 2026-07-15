"""Semantic, heading-aware chunking of a :class:`SourceDocument`.

The chunker splits ``content_markdown`` into sections along ATX headings (while
tracking a heading-path breadcrumb), packs each section's paragraphs/sentences
into token-bounded windows with configurable overlap, merges an undersized
trailing window back into its predecessor, and prepends a retrieval breadcrumb
to every chunk. It never splits mid-sentence unless a single sentence exceeds
the budget.
"""

from __future__ import annotations

import re

from dl_rag.config import Settings
from dl_rag.models.domain import Chunk, ChunkMetadata, SourceDocument
from dl_rag.utils.text import split_sentences
from dl_rag.utils.tokens import count_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class SemanticChunker:
    """Turn a document's markdown into ordered, retrieval-ready chunks."""

    def __init__(self, settings: Settings) -> None:
        self.max_tokens = max(1, settings.chunk_max_tokens)
        self.overlap = max(0, settings.chunk_overlap_tokens)
        self.min_tokens = max(0, settings.chunk_min_tokens)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chunk_document(self, doc: SourceDocument) -> list[Chunk]:
        markdown = (doc.content_markdown or "").strip()
        if not markdown:
            return []

        sections = self._split_sections(markdown, doc.title)
        if not sections:
            return []

        year = doc.issue_year or (
            doc.published_date.year if doc.published_date else None
        )
        month = doc.issue_month or (
            doc.published_date.strftime("%B") if doc.published_date else None
        )

        chunks: list[Chunk] = []
        index = 0
        for heading_path, body in sections:
            breadcrumb = self._breadcrumb(doc.title, heading_path)
            breadcrumb_tokens = count_tokens(breadcrumb)
            budget = self.max_tokens - breadcrumb_tokens
            if budget < max(1, self.min_tokens):
                budget = max(1, self.min_tokens)

            units = self._section_units(body, budget)
            if not units:
                continue

            windows = self._pack(units, budget)
            self._merge_trailing(windows)
            rendered = self._render_with_overlap(windows)

            for text in rendered:
                full_text = f"{breadcrumb}{text}" if breadcrumb else text
                metadata = ChunkMetadata(
                    url=doc.url,
                    title=doc.title,
                    category=doc.category,
                    content_type=doc.content_type,
                    year=year,
                    month=month,
                    author=doc.author,
                    issue=doc.issue_name,
                    published_date=doc.published_date,
                    tags=list(doc.tags),
                    entities=list(doc.entities),
                    heading_path=list(heading_path),
                )
                chunks.append(
                    Chunk(
                        id=Chunk.make_id(doc.id, index),
                        document_id=doc.id,
                        chunk_index=index,
                        text=full_text,
                        token_count=count_tokens(full_text),
                        metadata=metadata,
                    )
                )
                index += 1

        return chunks

    # ------------------------------------------------------------------ #
    # Sectioning
    # ------------------------------------------------------------------ #
    def _split_sections(
        self, markdown: str, title: str
    ) -> list[tuple[list[str], str]]:
        """Split markdown into ``(heading_path, body)`` sections.

        ``heading_path`` always starts with the document title; content headings
        extend/replace it by level. Headings inside fenced code are ignored.
        """
        sections: list[tuple[list[str], str]] = []
        heading_stack: list[tuple[int, str]] = [(0, title)]
        current: list[str] = []
        in_code = False
        title_norm = title.strip().lower()

        def flush() -> None:
            body = "\n".join(current).strip()
            if body:
                sections.append(([t for _, t in heading_stack], body))

        for raw_line in markdown.split("\n"):
            if _FENCE_RE.match(raw_line):
                in_code = not in_code
                current.append(raw_line)
                continue

            match = None if in_code else _HEADING_RE.match(raw_line)
            if match is None:
                current.append(raw_line)
                continue

            flush()
            current = []
            level = len(match.group(1))
            text = match.group(2).strip().rstrip("#").strip()
            while len(heading_stack) > 1 and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if text and text.lower() != title_norm:
                heading_stack.append((level, text))

        flush()
        return sections

    # ------------------------------------------------------------------ #
    # Unit extraction (paragraph → sentence → word)
    # ------------------------------------------------------------------ #
    def _section_units(self, body: str, budget: int) -> list[str]:
        units: list[str] = []
        for paragraph in self._paragraphs(body):
            if count_tokens(paragraph) <= budget:
                units.append(paragraph)
                continue
            sentences = split_sentences(paragraph) or [paragraph]
            for sentence in sentences:
                if count_tokens(sentence) <= budget:
                    units.append(sentence)
                else:
                    units.extend(self._split_by_words(sentence, budget))
        return units

    @staticmethod
    def _paragraphs(body: str) -> list[str]:
        return [p.strip() for p in _PARAGRAPH_SPLIT.split(body) if p.strip()]

    @staticmethod
    def _split_by_words(text: str, budget: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            current.append(word)
            if count_tokens(" ".join(current)) >= budget:
                pieces.append(" ".join(current))
                current = []
        if current:
            pieces.append(" ".join(current))
        return pieces or [text]

    # ------------------------------------------------------------------ #
    # Packing / overlap / merge
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pack(units: list[str], budget: int) -> list[list[str]]:
        windows: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for unit in units:
            unit_tokens = count_tokens(unit)
            if current and current_tokens + unit_tokens > budget:
                windows.append(current)
                current = []
                current_tokens = 0
            current.append(unit)
            current_tokens += unit_tokens
        if current:
            windows.append(current)
        return windows

    def _merge_trailing(self, windows: list[list[str]]) -> None:
        if len(windows) < 2:
            return
        last_tokens = sum(count_tokens(u) for u in windows[-1])
        if last_tokens < self.min_tokens:
            tail = windows.pop()
            windows[-1].extend(tail)

    def _render_with_overlap(self, windows: list[list[str]]) -> list[str]:
        rendered: list[str] = []
        for idx, window in enumerate(windows):
            text = "\n\n".join(window)
            if idx > 0 and self.overlap > 0:
                tail = self._overlap_tail(windows[idx - 1])
                if tail:
                    text = f"{tail}\n\n{text}"
            rendered.append(text)
        return rendered

    def _overlap_tail(self, units: list[str]) -> str:
        tail: list[str] = []
        tokens = 0
        for unit in reversed(units):
            tail.insert(0, unit)
            tokens += count_tokens(unit)
            if tokens >= self.overlap:
                break
        return "\n\n".join(tail)

    # ------------------------------------------------------------------ #
    # Breadcrumb
    # ------------------------------------------------------------------ #
    @staticmethod
    def _breadcrumb(title: str, heading_path: list[str]) -> str:
        sub = heading_path[1:] if heading_path and heading_path[0] == title else list(
            heading_path
        )
        sub = [s for s in sub if s]
        if title and sub:
            return f"{title} — {' > '.join(sub)}\n\n"
        if title:
            return f"{title}\n\n"
        if sub:
            return f"{' > '.join(sub)}\n\n"
        return ""
