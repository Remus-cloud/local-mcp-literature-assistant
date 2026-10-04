"""Tests for chunking and local BM25 retrieval."""

import pytest

from literature_bot.models import PaperPage
from literature_bot.retrieval import (
    build_paper_chunks,
    retrieve_relevant_chunks,
    split_page_text,
    tokenize_for_retrieval,
)


def test_split_page_text_creates_overlap():
    text = "".join(str(index % 10) for index in range(160))

    chunks = split_page_text(text, max_characters=100, overlap_characters=20)

    assert len(chunks) == 2
    assert len(chunks[0]) <= 100
    assert chunks[0][-20:] in chunks[1]


@pytest.mark.parametrize(
    ("maximum", "overlap"),
    [(0, 0), (100, -1), (100, 100)],
)
def test_split_page_text_rejects_invalid_limits(maximum, overlap):
    with pytest.raises(ValueError):
        split_page_text("text", maximum, overlap)


def test_build_chunks_preserves_source_page():
    pages = [
        PaperPage(1, "first page evidence", 19),
        PaperPage(2, "second page evidence", 20),
    ]

    chunks = build_paper_chunks(pages, max_characters=100, overlap_characters=10)

    assert [chunk.chunk_id for chunk in chunks] == [1, 2]
    assert [chunk.page_number for chunk in chunks] == [1, 2]


def test_bm25_ranks_matching_chunk_first():
    pages = [
        PaperPage(1, "transformer language generation attention", 41),
        PaperPage(2, "object detection bounding boxes vision", 38),
    ]
    chunks = build_paper_chunks(pages, max_characters=100, overlap_characters=10)

    hits = retrieve_relevant_chunks("object detection", chunks, top_k=1)

    assert hits[0].chunk.page_number == 2
    assert hits[0].score > 0


def test_tokenizer_removes_common_stopwords():
    assert tokenize_for_retrieval("How is the object detected?") == [
        "object",
        "detected",
    ]
