"""Stateful, model-free services used by the MCP tool layer."""

from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path

import anyio

from ..arxiv_service import create_arxiv_client, search_arxiv
from ..models import PaperMetadata, PaperPage, TextChunk
from ..pdf_service import download_paper_pdf, extract_pdf_pages
from ..retrieval import build_paper_chunks, retrieve_relevant_chunks
from ..scientific_retrieval import Specter2PaperRetriever
from .schemas import (
    ArxivIdInput,
    EvidenceItem,
    EvidenceResult,
    PaperDetailsResult,
    PaperRecord,
    PreparePaperResult,
    RetrieveEvidenceInput,
    SearchPapersInput,
    SearchPapersResult,
)


EVIDENCE_CHARACTER_LIMIT = 24_000
EVIDENCE_CHUNK_LIMIT = 3_000


def default_papers_directory() -> Path:
    """Return a cross-platform papers directory with an environment override."""

    configured = os.getenv("LITERATURE_BOT_PAPERS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "papers"


@dataclass(frozen=True)
class PreparedPaper:
    metadata: PaperMetadata
    pdf_path: Path
    pages: list[PaperPage]
    chunks: list[TextChunk]


class LiteratureService:
    """Manage paper metadata plus process-local page and chunk caches."""

    def __init__(
        self,
        papers_directory: str | Path | None = None,
        *,
        paper_retriever=None,
    ) -> None:
        self.papers_directory = Path(
            papers_directory or default_papers_directory()
        ).resolve()
        self._arxiv_client = create_arxiv_client()
        self._papers: dict[str, PaperMetadata] = {}
        self._prepared: dict[str, PreparedPaper] = {}
        self._paper_retriever = paper_retriever
        self._paper_retriever_mode = os.getenv(
            "LITERATURE_BOT_PAPER_RETRIEVER", "specter2"
        ).strip().lower()
        if self._paper_retriever_mode not in {"arxiv", "specter2"}:
            raise ValueError(
                "LITERATURE_BOT_PAPER_RETRIEVER must be 'arxiv' or 'specter2'"
            )
        self._specter2_candidate_pool = self._candidate_pool_size()

    @staticmethod
    def _candidate_pool_size() -> int:
        raw = os.getenv("LITERATURE_BOT_SPECTER2_CANDIDATE_POOL", "10")
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(
                "LITERATURE_BOT_SPECTER2_CANDIDATE_POOL must be an integer"
            ) from error
        if not 1 <= value <= 10:
            raise ValueError(
                "LITERATURE_BOT_SPECTER2_CANDIDATE_POOL must be between 1 and 10"
            )
        return value

    def _get_paper_retriever(self):
        if self._paper_retriever is None:
            self._paper_retriever = Specter2PaperRetriever()
        return self._paper_retriever

    def _rank_papers(
        self,
        query: str,
        papers: list[PaperMetadata],
        top_k: int,
    ) -> list[PaperMetadata]:
        ranked = self._get_paper_retriever().rank(query, papers, top_k=top_k)
        return [hit.paper for hit in ranked]

    @staticmethod
    def _key(arxiv_id: str) -> str:
        return arxiv_id.strip().lower()

    async def search_papers(self, params: SearchPapersInput) -> SearchPapersResult:
        candidate_count = params.max_results
        if self._paper_retriever_mode == "specter2":
            candidate_count = max(candidate_count, self._specter2_candidate_pool)
        papers = await anyio.to_thread.run_sync(
            partial(
                search_arxiv,
                query=params.query,
                max_results=candidate_count,
                client=self._arxiv_client,
            )
        )

        actual_candidate_count = len(papers)
        if self._paper_retriever_mode == "specter2" and papers:
            ranked = await anyio.to_thread.run_sync(
                partial(
                    self._rank_papers,
                    params.query,
                    papers,
                    top_k=min(params.max_results, len(papers)),
                )
            )
            papers = ranked

        for paper in papers:
            self._papers[self._key(paper.arxiv_id)] = paper

        return SearchPapersResult(
            query=params.query,
            count=len(papers),
            candidate_count=actual_candidate_count,
            retrieval_mode=self._paper_retriever_mode,
            reranking_applied=self._paper_retriever_mode == "specter2" and bool(papers),
            papers=[
                PaperRecord.from_metadata(paper, summary_limit=1_000)
                for paper in papers
            ],
        )

    async def _resolve_paper(self, arxiv_id: str) -> PaperMetadata:
        key = self._key(arxiv_id)
        cached = self._papers.get(key)
        if cached is not None:
            return cached

        papers = await anyio.to_thread.run_sync(
            partial(
                search_arxiv,
                query=f"id:{arxiv_id}",
                max_results=1,
                client=self._arxiv_client,
            )
        )
        if not papers:
            raise LookupError(
                f"No arXiv paper was found for '{arxiv_id}'. "
                "Call literature_search_papers to obtain a valid arXiv ID."
            )

        paper = papers[0]
        self._papers[self._key(paper.arxiv_id)] = paper
        self._papers[key] = paper
        return paper

    async def get_paper_details(
        self,
        params: ArxivIdInput,
    ) -> PaperDetailsResult:
        paper = await self._resolve_paper(params.arxiv_id)
        return PaperDetailsResult(
            paper=PaperRecord.from_metadata(paper, summary_limit=4_000),
            prepared_in_memory=self._key(params.arxiv_id) in self._prepared,
        )

    async def prepare_paper(
        self,
        params: ArxivIdInput,
    ) -> PreparePaperResult:
        key = self._key(params.arxiv_id)
        cached = self._prepared.get(key)
        if cached is not None:
            return self._prepare_result(cached, reused_memory_cache=True)

        paper = await self._resolve_paper(params.arxiv_id)
        pdf_path = await anyio.to_thread.run_sync(
            partial(
                download_paper_pdf,
                paper=paper,
                output_directory=self.papers_directory,
            )
        )
        pages = await anyio.to_thread.run_sync(extract_pdf_pages, pdf_path)
        chunks = await anyio.to_thread.run_sync(build_paper_chunks, pages)

        if not chunks:
            raise ValueError(
                "The PDF contained no searchable text. It may be scanned or "
                "image-only; OCR is not supported in this version."
            )

        prepared = PreparedPaper(
            metadata=paper,
            pdf_path=pdf_path,
            pages=pages,
            chunks=chunks,
        )
        self._prepared[key] = prepared
        self._prepared[self._key(paper.arxiv_id)] = prepared
        return self._prepare_result(prepared, reused_memory_cache=False)

    @staticmethod
    def _prepare_result(
        prepared: PreparedPaper,
        *,
        reused_memory_cache: bool,
    ) -> PreparePaperResult:
        return PreparePaperResult(
            arxiv_id=prepared.metadata.arxiv_id,
            title=prepared.metadata.title,
            pdf_path=str(prepared.pdf_path),
            page_count=len(prepared.pages),
            nonempty_page_count=sum(
                page.character_count > 0 for page in prepared.pages
            ),
            chunk_count=len(prepared.chunks),
            reused_memory_cache=reused_memory_cache,
        )

    async def retrieve_evidence(
        self,
        params: RetrieveEvidenceInput,
    ) -> EvidenceResult:
        key = self._key(params.arxiv_id)
        auto_prepared = key not in self._prepared
        if auto_prepared:
            await self.prepare_paper(ArxivIdInput(arxiv_id=params.arxiv_id))

        prepared = self._prepared[key]
        top_k = min(params.top_k, len(prepared.chunks))
        hits = await anyio.to_thread.run_sync(
            partial(
                retrieve_relevant_chunks,
                query=params.query,
                chunks=prepared.chunks,
                top_k=top_k,
            )
        )

        evidence: list[EvidenceItem] = []
        used_characters = 0
        for rank, hit in enumerate(hits, start=1):
            remaining = EVIDENCE_CHARACTER_LIMIT - used_characters
            if remaining <= 0:
                break
            text = hit.chunk.text[: min(EVIDENCE_CHUNK_LIMIT, remaining)]
            evidence.append(
                EvidenceItem(
                    rank=rank,
                    page_number=hit.chunk.page_number,
                    chunk_id=hit.chunk.chunk_id,
                    score=round(hit.score, 6),
                    text=text,
                )
            )
            used_characters += len(text)

        return EvidenceResult(
            arxiv_id=prepared.metadata.arxiv_id,
            title=prepared.metadata.title,
            query=params.query,
            evidence_count=len(evidence),
            auto_prepared=auto_prepared,
            evidence=evidence,
        )

    async def get_metadata_resource(self, arxiv_id: str) -> dict:
        """Return JSON-ready metadata for an application-controlled resource."""

        params = ArxivIdInput(arxiv_id=arxiv_id)
        details = await self.get_paper_details(params)
        return details.model_dump(mode="json")

    async def get_page_resource(
        self,
        arxiv_id: str,
        page_number: int,
    ) -> dict:
        """Read one prepared page without causing a PDF download."""

        params = ArxivIdInput(arxiv_id=arxiv_id)
        if page_number < 1:
            raise ValueError("page_number must be at least 1")

        prepared = self._prepared.get(self._key(params.arxiv_id))
        if prepared is None:
            raise LookupError(
                f"Paper '{params.arxiv_id}' is not prepared in memory. "
                "Call literature_prepare_paper before reading page resources."
            )

        if page_number > len(prepared.pages):
            raise ValueError(
                f"page_number must be between 1 and {len(prepared.pages)} "
                f"for paper '{prepared.metadata.arxiv_id}'."
            )

        page = prepared.pages[page_number - 1]
        return {
            "arxiv_id": prepared.metadata.arxiv_id,
            "title": prepared.metadata.title,
            "page_number": page.page_number,
            "character_count": page.character_count,
            "text": page.text,
        }
