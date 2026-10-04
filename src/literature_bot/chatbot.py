"""High-level orchestration for one stateless paper question."""

from openai import OpenAI

from .llm_service import answer_question_from_paper, create_retrieval_query
from .models import PaperMetadata, TextChunk
from .retrieval import retrieve_relevant_chunks


def ask_paper_bot(
    question: str,
    paper: PaperMetadata,
    chunks: list[TextChunk],
    *,
    client: OpenAI,
    model: str,
    top_k: int = 6,
) -> str:
    """Complete one independent retrieval-and-answer cycle."""

    retrieval_query = create_retrieval_query(
        question,
        client=client,
        model=model,
    )
    hits = retrieve_relevant_chunks(
        query=retrieval_query,
        chunks=chunks,
        top_k=top_k,
    )
    return answer_question_from_paper(
        question=question,
        paper=paper,
        retrieval_hits=hits,
        client=client,
        model=model,
    )
