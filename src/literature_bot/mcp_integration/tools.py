"""Register model-free literature workflows as MCP tools."""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field

from .schemas import (
    ArxivIdInput,
    EvidenceResult,
    PaperDetailsResult,
    PreparePaperResult,
    RetrieveEvidenceInput,
    SearchPapersInput,
    SearchPapersResult,
)
from .service import LiteratureService


def register_tools(server: MCPServer, service: LiteratureService) -> None:
    """Attach the four literature workflow tools to an MCP server."""

    @server.tool(
        name="literature_search_papers",
        title="Search arXiv Papers",
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )
    async def search_papers_tool(
        query: Annotated[
            str,
            Field(
                min_length=2,
                max_length=500,
                description="English arXiv search query.",
            ),
        ],
        max_results: Annotated[
            int,
            Field(ge=1, le=10, description="Return between 1 and 10 papers."),
        ] = 5,
    ) -> SearchPapersResult:
        """Search arXiv and return concise paper metadata.

        Use this first when the user wants literature recommendations or has not
        supplied an arXiv ID. It does not download PDFs. The returned arXiv IDs
        can be passed to the other literature tools.
        """

        params = SearchPapersInput(query=query, max_results=max_results)
        return await service.search_papers(params)

    @server.tool(
        name="literature_get_paper_details",
        title="Get arXiv Paper Details",
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )
    async def get_paper_details_tool(
        arxiv_id: Annotated[
            str,
            Field(
                min_length=5,
                max_length=64,
                description="arXiv ID such as '2508.19294v2'.",
            ),
        ],
    ) -> PaperDetailsResult:
        """Return detailed metadata for one arXiv paper.

        Use this for title, authors, abstract, categories, dates, and URLs. If
        the paper has not been seen in this session, the server resolves the ID
        through arXiv. It does not download the PDF.
        """

        return await service.get_paper_details(ArxivIdInput(arxiv_id=arxiv_id))

    @server.tool(
        name="literature_prepare_paper",
        title="Prepare Paper for Retrieval",
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )
    async def prepare_paper_tool(
        arxiv_id: Annotated[
            str,
            Field(
                min_length=5,
                max_length=64,
                description="arXiv ID returned by literature_search_papers.",
            ),
        ],
    ) -> PreparePaperResult:
        """Download, extract, and chunk one paper for evidence retrieval.

        The PDF is cached in the configured local papers directory and is not
        overwritten. Extracted pages and chunks stay only in server memory.
        Repeating the call is safe and reuses the cache.
        """

        return await service.prepare_paper(ArxivIdInput(arxiv_id=arxiv_id))

    @server.tool(
        name="literature_retrieve_paper_evidence",
        title="Retrieve Evidence from a Paper",
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )
    async def retrieve_paper_evidence_tool(
        arxiv_id: Annotated[
            str,
            Field(
                min_length=5,
                max_length=64,
                description="arXiv ID identifying the paper to search.",
            ),
        ],
        query: Annotated[
            str,
            Field(
                min_length=2,
                max_length=500,
                description=(
                    "Concise English keywords. Translate non-English questions "
                    "before calling."
                ),
            ),
        ],
        top_k: Annotated[
            int,
            Field(ge=1, le=8, description="Return between 1 and 8 chunks."),
        ] = 6,
    ) -> EvidenceResult:
        """Retrieve page-cited evidence chunks from one paper.

        Use this before answering a question about paper content. The query must
        be concise English keywords because retrieval uses local BM25 rather
        than a model. If necessary, the paper is prepared automatically. Return
        the page numbers with any answer built from these results.
        """

        params = RetrieveEvidenceInput(
            arxiv_id=arxiv_id,
            query=query,
            top_k=top_k,
        )
        return await service.retrieve_evidence(params)
