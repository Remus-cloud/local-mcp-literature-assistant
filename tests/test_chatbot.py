"""Tests for the stateless chatbot orchestration layer."""

from unittest.mock import Mock

from literature_bot import chatbot
from literature_bot.models import PaperMetadata, RetrievalHit, TextChunk


def test_ask_paper_bot_connects_query_retrieval_and_answer(monkeypatch):
    paper = PaperMetadata(
        arxiv_id="2601.00001v1",
        title="Test Paper",
        authors=["Ada Researcher"],
        summary="Summary",
        published="2026-01-02",
        updated="2026-01-02",
        entry_url="https://arxiv.org/abs/2601.00001v1",
        pdf_url="https://arxiv.org/pdf/2601.00001v1",
        categories=["cs.AI"],
    )
    chunk = TextChunk(1, 3, "relevant evidence", 17)
    hit = RetrievalHit(chunk=chunk, score=1.0)
    client = Mock()

    create_query = Mock(return_value="relevant keywords")
    retrieve = Mock(return_value=[hit])
    answer = Mock(return_value="grounded answer")
    monkeypatch.setattr(chatbot, "create_retrieval_query", create_query)
    monkeypatch.setattr(chatbot, "retrieve_relevant_chunks", retrieve)
    monkeypatch.setattr(chatbot, "answer_question_from_paper", answer)

    result = chatbot.ask_paper_bot(
        "用户问题",
        paper,
        [chunk],
        client=client,
        model="qwen-test",
        top_k=1,
    )

    assert result == "grounded answer"
    create_query.assert_called_once_with(
        "用户问题",
        client=client,
        model="qwen-test",
    )
    retrieve.assert_called_once_with(
        query="relevant keywords",
        chunks=[chunk],
        top_k=1,
    )
    answer.assert_called_once_with(
        question="用户问题",
        paper=paper,
        retrieval_hits=[hit],
        client=client,
        model="qwen-test",
    )
