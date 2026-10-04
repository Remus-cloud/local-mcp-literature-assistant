"""Tests for paper preparation state and evidence limits."""

import asyncio
from pathlib import Path

from literature_bot.mcp_integration import service as service_module
from literature_bot.mcp_integration.schemas import (
    ArxivIdInput,
    RetrieveEvidenceInput,
    SearchPapersInput,
)
from literature_bot.mcp_integration.service import LiteratureService
from literature_bot.models import PaperMetadata, PaperPage


def make_paper() -> PaperMetadata:
    return PaperMetadata(
        arxiv_id="2601.00001v1",
        title="Test Paper",
        authors=["Ada Researcher"],
        summary="Summary",
        published="2026-01-01",
        updated="2026-01-02",
        entry_url="https://arxiv.org/abs/2601.00001v1",
        pdf_url="https://arxiv.org/pdf/2601.00001v1",
        categories=["cs.AI"],
    )


def test_service_prepares_once_and_retrieves_page_evidence(tmp_path, monkeypatch):
    calls = {"download": 0}

    def fake_search(**kwargs):
        return [make_paper()]

    def fake_download(**kwargs):
        calls["download"] += 1
        return Path(tmp_path / "paper.pdf")

    monkeypatch.setattr(service_module, "search_arxiv", fake_search)
    monkeypatch.setattr(service_module, "download_paper_pdf", fake_download)
    monkeypatch.setattr(
        service_module,
        "extract_pdf_pages",
        lambda path: [
            PaperPage(1, "language model generation", 25),
            PaperPage(2, "object detection bounding boxes", 31),
        ],
    )

    async def scenario():
        service = LiteratureService(tmp_path)
        searched = await service.search_papers(
            SearchPapersInput(query="object detection", max_results=1)
        )
        assert searched.count == 1

        first = await service.prepare_paper(
            ArxivIdInput(arxiv_id="2601.00001v1")
        )
        second = await service.prepare_paper(
            ArxivIdInput(arxiv_id="2601.00001v1")
        )
        assert first.reused_memory_cache is False
        assert second.reused_memory_cache is True
        assert calls["download"] == 1

        evidence = await service.retrieve_evidence(
            RetrieveEvidenceInput(
                arxiv_id="2601.00001v1",
                query="object detection",
                top_k=1,
            )
        )
        assert evidence.auto_prepared is False
        assert evidence.evidence[0].page_number == 2

    asyncio.run(scenario())
