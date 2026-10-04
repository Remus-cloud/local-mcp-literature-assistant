"""Reusable business logic for the literature search and Q&A assistant."""

from .arxiv_service import choose_paper, create_arxiv_client, search_arxiv
from .chatbot import ask_paper_bot
from .config import Settings, load_settings
from .llm_service import (
    answer_question_from_paper,
    build_evidence_text,
    create_llm_client,
    create_retrieval_query,
)
from .models import PaperMetadata, PaperPage, RetrievalHit, TextChunk
from .pdf_service import download_paper_pdf, extract_pdf_pages
from .retrieval import build_paper_chunks, retrieve_relevant_chunks

__all__ = [
    "PaperMetadata",
    "PaperPage",
    "RetrievalHit",
    "Settings",
    "TextChunk",
    "answer_question_from_paper",
    "ask_paper_bot",
    "build_evidence_text",
    "build_paper_chunks",
    "choose_paper",
    "create_arxiv_client",
    "create_llm_client",
    "create_retrieval_query",
    "download_paper_pdf",
    "extract_pdf_pages",
    "load_settings",
    "retrieve_relevant_chunks",
    "search_arxiv",
]
