"""Split paper text and retrieve relevant chunks with BM25."""

from collections import Counter
import math
import re

from .models import PaperPage, RetrievalHit, TextChunk


ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
}


def split_page_text(
    text: str,
    max_characters: int = 1800,
    overlap_characters: int = 250,
) -> list[str]:
    """Split one page into overlapping chunks near natural boundaries."""

    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero.")

    if not 0 <= overlap_characters < max_characters:
        raise ValueError(
            "overlap_characters must be non-negative and smaller than "
            "max_characters."
        )

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        proposed_end = min(start + max_characters, len(text))
        end = proposed_end

        if proposed_end < len(text):
            search_start = start + max_characters // 2
            newline_boundary = text.rfind("\n", search_start, proposed_end)
            sentence_boundary = text.rfind(". ", search_start, proposed_end)
            best_boundary = max(newline_boundary, sentence_boundary)

            if best_boundary > start:
                end = best_boundary + 1

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(text):
            break

        next_start = end - overlap_characters
        start = next_start if next_start > start else end

    return chunks


def build_paper_chunks(
    pages: list[PaperPage],
    max_characters: int = 1800,
    overlap_characters: int = 250,
) -> list[TextChunk]:
    """Convert all extracted pages into page-aware chunks."""

    chunks: list[TextChunk] = []

    for page in pages:
        for chunk_text in split_page_text(
            page.text,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        ):
            chunks.append(
                TextChunk(
                    chunk_id=len(chunks) + 1,
                    page_number=page.page_number,
                    text=chunk_text,
                    character_count=len(chunk_text),
                )
            )

    return chunks


def tokenize_for_retrieval(text: str) -> list[str]:
    """Tokenize English academic text for local keyword retrieval."""

    raw_tokens = re.findall(
        r"[a-zA-Z][a-zA-Z0-9'-]*|\d+(?:\.\d+)?",
        text.lower(),
    )
    return [
        token
        for token in raw_tokens
        if token not in ENGLISH_STOPWORDS and len(token) > 1
    ]


def retrieve_relevant_chunks(
    query: str,
    chunks: list[TextChunk],
    top_k: int = 6,
) -> list[RetrievalHit]:
    """Rank chunks with the BM25 formula used by the verified notebook."""

    if not chunks:
        raise ValueError("The chunk list cannot be empty.")

    if not 1 <= top_k <= len(chunks):
        raise ValueError(f"top_k must be between 1 and {len(chunks)}.")

    query_tokens = tokenize_for_retrieval(query)

    if not query_tokens:
        raise ValueError("The query contains no usable English keywords.")

    document_tokens = [tokenize_for_retrieval(chunk.text) for chunk in chunks]
    document_frequencies: Counter[str] = Counter()

    for tokens in document_tokens:
        document_frequencies.update(set(tokens))

    document_count = len(chunks)
    average_document_length = (
        sum(len(tokens) for tokens in document_tokens) / document_count
    )

    if average_document_length == 0:
        raise ValueError("The chunks contain no usable English tokens.")

    query_counts = Counter(query_tokens)
    k1 = 1.5
    b = 0.75
    scored_chunks: list[RetrievalHit] = []

    for chunk, tokens in zip(chunks, document_tokens):
        token_counts = Counter(tokens)
        document_length = len(tokens)
        score = 0.0

        for token, query_frequency in query_counts.items():
            term_frequency = token_counts.get(token, 0)

            if term_frequency == 0:
                continue

            document_frequency = document_frequencies[token]
            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_normalization = (
                1 - b + b * document_length / average_document_length
            )
            term_score = (
                inverse_document_frequency
                * term_frequency
                * (k1 + 1)
                / (term_frequency + k1 * length_normalization)
            )
            score += term_score * (1 + math.log(query_frequency))

        scored_chunks.append(RetrievalHit(chunk=chunk, score=score))

    scored_chunks.sort(key=lambda hit: hit.score, reverse=True)
    return scored_chunks[:top_k]
