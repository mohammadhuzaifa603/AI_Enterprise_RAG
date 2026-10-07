"""
Chunking.

Splits per-page text into overlapping character-window chunks. Page
provenance is never lost: every chunk carries the page_number it came
from, which is required for citations later in the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.config import settings


@dataclass
class ChunkCandidate:
    page_number: int
    chunk_index: int
    text: str


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence-ish splitter; avoids pulling in a heavy NLP dep.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_page_text(
    page_number: int,
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[ChunkCandidate]:
    """Chunk a single page's text into overlapping windows.

    Uses a sentence-aware greedy packer: sentences are accumulated until
    the chunk would exceed `chunk_size`, then a new chunk starts,
    carrying `overlap` characters of trailing context forward.
    """
    chunk_size = chunk_size or settings.chunk_size_chars
    overlap = overlap or settings.chunk_overlap_chars
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            # carry overlap forward
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail} {sentence}".strip()
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    # Fallback: single very long "sentence" with no punctuation at all.
    if not chunks:
        for start in range(0, len(text), chunk_size - overlap or chunk_size):
            chunks.append(text[start : start + chunk_size])

    return [
        ChunkCandidate(page_number=page_number, chunk_index=i, text=c)
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def chunk_document_pages(pages: List[tuple[int, str]]) -> List[ChunkCandidate]:
    """Chunk every page of a document; chunk_index is unique per document."""
    all_chunks: List[ChunkCandidate] = []
    running_index = 0
    for page_number, text in pages:
        page_chunks = chunk_page_text(page_number, text)
        for pc in page_chunks:
            all_chunks.append(ChunkCandidate(page_number=page_number, chunk_index=running_index, text=pc.text))
            running_index += 1
    return all_chunks
