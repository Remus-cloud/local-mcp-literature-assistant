"""Core data models used by the literature assistant."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperMetadata:
    """Basic metadata for one arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str
    entry_url: str
    pdf_url: str
    categories: list[str]


@dataclass(frozen=True)
class PaperPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str
    character_count: int


@dataclass(frozen=True)
class TextChunk:
    """A page-aware chunk of paper text."""

    chunk_id: int
    page_number: int
    text: str
    character_count: int


@dataclass(frozen=True)
class RetrievalHit:
    """A retrieved text chunk and its relevance score."""

    chunk: TextChunk
    score: float
