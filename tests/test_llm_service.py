"""Tests for Qwen request construction without making API calls."""

from types import SimpleNamespace

import pytest

from literature_bot.llm_service import (
    answer_question_from_paper,
    build_evidence_text,
    create_retrieval_query,
)
from literature_bot.models import PaperMetadata, RetrievalHit, TextChunk


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, content: str):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def make_paper() -> PaperMetadata:
    return PaperMetadata(
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


def make_hit() -> RetrievalHit:
    chunk = TextChunk(1, 7, "The reported accuracy is 90 percent.", 36)
    return RetrievalHit(chunk=chunk, score=2.5)


def test_create_retrieval_query_uses_qwen_compatible_options():
    client = FakeClient("object detection multimodal model")

    result = create_retrieval_query(
        "这篇论文如何完成目标检测？",
        client=client,
        model="qwen-test",
    )

    request = client.completions.calls[0]
    assert result == "object detection multimodal model"
    assert request["model"] == "qwen-test"
    assert request["extra_body"] == {"enable_thinking": False}


def test_build_evidence_text_includes_page_and_chunk_ids():
    evidence = build_evidence_text([make_hit()])

    assert "论文第 7 页" in evidence
    assert "文本块 1" in evidence


def test_answer_question_includes_evidence_and_returns_content():
    client = FakeClient("实验准确率为 90%。[第 7 页]")

    answer = answer_question_from_paper(
        question="准确率是多少？",
        paper=make_paper(),
        retrieval_hits=[make_hit()],
        client=client,
        model="qwen-test",
    )

    request = client.completions.calls[0]
    assert answer == "实验准确率为 90%。[第 7 页]"
    assert "论文第 7 页" in request["messages"][1]["content"]


def test_answer_question_rejects_missing_evidence():
    with pytest.raises(ValueError, match="no paper evidence"):
        answer_question_from_paper(
            question="问题",
            paper=make_paper(),
            retrieval_hits=[],
            client=FakeClient("unused"),
            model="qwen-test",
        )
