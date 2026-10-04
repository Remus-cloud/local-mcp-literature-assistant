"""Register application-controlled literature resources."""

from mcp.server.mcpserver import MCPServer

from .service import LiteratureService


GUIDE_TEXT = """# Literature MCP 使用指南

本服务器提供 arXiv 搜索、PDF 准备和论文证据检索。

## 推荐流程

1. 使用 `literature_search_papers` 搜索论文并取得 arXiv ID。
2. 只需要摘要和元数据时，使用 `literature_get_paper_details`。
3. 需要正文内容时，使用 `literature_prepare_paper` 准备 PDF。
4. 使用 `literature_retrieve_paper_evidence` 检索相关证据。
5. 回答中的关键结论必须使用 `[第 X 页]` 标注来源。

## Resources

- `literature://guide`：本指南。
- `literature://papers/{arxiv_id}/metadata`：论文元数据。
- `literature://papers/{arxiv_id}/pages/{page_number}`：已准备论文的指定页面。

页面 Resource 不会自动下载论文。若论文尚未准备，请先调用
`literature_prepare_paper`。
""".strip()


def register_resources(server: MCPServer, service: LiteratureService) -> None:
    """Attach one fixed resource and two resource templates."""

    @server.resource(
        "literature://guide",
        name="literature_guide",
        title="Literature MCP Guide",
        description="Workflow and citation guidance for this literature server.",
        mime_type="text/markdown",
    )
    def literature_guide() -> str:
        """Return the server workflow and evidence citation guide."""

        return GUIDE_TEXT

    @server.resource(
        "literature://papers/{arxiv_id}/metadata",
        name="paper_metadata",
        title="arXiv Paper Metadata",
        description=(
            "Read title, authors, abstract, dates, categories, and URLs for an "
            "arXiv paper."
        ),
        mime_type="application/json",
    )
    async def paper_metadata(arxiv_id: str) -> dict:
        """Return normalized metadata for one arXiv identifier."""

        return await service.get_metadata_resource(arxiv_id)

    @server.resource(
        "literature://papers/{arxiv_id}/pages/{page_number}",
        name="paper_page",
        title="Prepared Paper Page",
        description=(
            "Read one page from a paper already prepared in this MCP server "
            "process."
        ),
        mime_type="application/json",
    )
    async def paper_page(arxiv_id: str, page_number: int) -> dict:
        """Return page text without automatically downloading or parsing a PDF."""

        return await service.get_page_resource(arxiv_id, page_number)
