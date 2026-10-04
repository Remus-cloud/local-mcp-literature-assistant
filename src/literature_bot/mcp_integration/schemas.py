"""Validated input and structured output models for MCP tools."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import PaperMetadata


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and trims strings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchPapersInput(StrictModel):
    query: str = Field(
        min_length=2,
        max_length=500,
        description=(
            "English arXiv search query, for example "
            "'multimodal object detection'."
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of papers to return, from 1 to 10.",
    )


class ArxivIdInput(StrictModel):
    arxiv_id: str = Field(
        min_length=5,
        max_length=64,
        pattern=r"^[A-Za-z0-9./-]+$",
        description=(
            "An arXiv identifier returned by literature_search_papers, "
            "for example '2508.19294v2'."
        ),
    )

    @field_validator("arxiv_id")
    @classmethod
    def reject_path_traversal(cls, value: str) -> str:
        if ".." in value or value.startswith(("/", ".")):
            raise ValueError("arxiv_id must be an identifier, not a file path")
        return value


class RetrieveEvidenceInput(ArxivIdInput):
    query: str = Field(
        min_length=2,
        max_length=500,
        description=(
            "Concise English keywords describing the evidence to find inside "
            "the paper. Translate a non-English user question before calling."
        ),
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=8,
        description="Number of relevant evidence chunks to return, from 1 to 8.",
    )


class PaperRecord(StrictModel):
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str
    entry_url: str
    pdf_url: str
    categories: list[str]

    @classmethod
    def from_metadata(
        cls,
        paper: PaperMetadata,
        *,
        summary_limit: int | None = None,
    ) -> "PaperRecord":
        summary = paper.summary
        if summary_limit is not None and len(summary) > summary_limit:
            summary = summary[:summary_limit].rstrip() + "..."
        return cls(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            summary=summary,
            published=paper.published,
            updated=paper.updated,
            entry_url=paper.entry_url,
            pdf_url=paper.pdf_url,
            categories=paper.categories,
        )


class SearchPapersResult(StrictModel):
    query: str
    count: int
    papers: list[PaperRecord]


class PaperDetailsResult(StrictModel):
    paper: PaperRecord
    prepared_in_memory: bool


class PreparePaperResult(StrictModel):
    arxiv_id: str
    title: str
    pdf_path: str
    page_count: int
    nonempty_page_count: int
    chunk_count: int
    reused_memory_cache: bool


class EvidenceItem(StrictModel):
    rank: int
    page_number: int
    chunk_id: int
    score: float
    text: str


class EvidenceResult(StrictModel):
    arxiv_id: str
    title: str
    query: str
    evidence_count: int
    auto_prepared: bool
    evidence: list[EvidenceItem]
