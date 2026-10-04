"""In-process MCP discovery and invocation tests."""

import asyncio

from mcp import Client

from literature_bot.mcp_integration.schemas import (
    PaperRecord,
    SearchPapersResult,
)
from literature_bot.mcp_integration.server import create_server


class FakeLiteratureService:
    async def search_papers(self, params):
        return SearchPapersResult(
            query=params.query,
            count=1,
            papers=[
                PaperRecord(
                    arxiv_id="2601.00001v1",
                    title="Test Paper",
                    authors=["Ada Researcher"],
                    summary="Test summary",
                    published="2026-01-01",
                    updated="2026-01-02",
                    entry_url="https://arxiv.org/abs/2601.00001v1",
                    pdf_url="https://arxiv.org/pdf/2601.00001v1",
                    categories=["cs.AI"],
                )
            ],
        )

    async def get_paper_details(self, params):
        raise AssertionError("not used")

    async def prepare_paper(self, params):
        raise AssertionError("not used")

    async def retrieve_evidence(self, params):
        raise AssertionError("not used")


def test_server_lists_four_tools_and_calls_search():
    async def scenario():
        server = create_server(service=FakeLiteratureService())
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == {
                "literature_search_papers",
                "literature_get_paper_details",
                "literature_prepare_paper",
                "literature_retrieve_paper_evidence",
            }

            result = await client.call_tool(
                "literature_search_papers",
                {"query": "multimodal models", "max_results": 1},
            )
            assert result.is_error is False
            assert result.structured_content["count"] == 1
            assert result.structured_content["papers"][0]["arxiv_id"] == (
                "2601.00001v1"
            )

    asyncio.run(scenario())
