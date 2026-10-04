"""Local stdio MCP server for literature search and paper evidence."""

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .service import LiteratureService
from .prompts import register_prompts
from .resources import register_resources
from .tools import register_tools


def create_server(
    *,
    papers_directory: str | Path | None = None,
    service: LiteratureService | None = None,
) -> MCPServer:
    """Build a server; dependency injection keeps it testable in-process."""

    server = MCPServer(
        name="literature_mcp",
        title="Literature Search MCP",
        description=(
            "Search arXiv, prepare local PDFs, and retrieve page-cited paper "
            "evidence. This server contains no language model."
        ),
        instructions=(
            "Search before choosing a paper. Use concise English keywords for "
            "evidence retrieval. Base answers only on returned evidence and cite "
            "its page_number fields."
        ),
        version="0.2.0",
    )
    active_service = service or LiteratureService(papers_directory)
    register_tools(server, active_service)
    register_resources(server, active_service)
    register_prompts(server)
    return server


mcp = create_server()


def main() -> None:
    """Run over stdio; never print application logs to stdout."""

    mcp.run("stdio")


if __name__ == "__main__":
    main()
