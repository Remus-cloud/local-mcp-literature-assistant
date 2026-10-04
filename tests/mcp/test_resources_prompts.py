"""In-process tests for MCP resources and prompt templates."""

import asyncio
import json

from mcp import Client
from mcp_types import TextContent, TextResourceContents

from literature_bot.mcp_integration.server import create_server


class FakeResourceService:
    async def search_papers(self, params):
        raise AssertionError("not used")

    async def get_paper_details(self, params):
        raise AssertionError("not used")

    async def prepare_paper(self, params):
        raise AssertionError("not used")

    async def retrieve_evidence(self, params):
        raise AssertionError("not used")

    async def get_metadata_resource(self, arxiv_id):
        return {"paper": {"arxiv_id": arxiv_id, "title": "Test Paper"}}

    async def get_page_resource(self, arxiv_id, page_number):
        return {
            "arxiv_id": arxiv_id,
            "page_number": page_number,
            "text": "Evidence from the requested page.",
        }


def test_server_lists_and_reads_resources_and_prompts():
    async def scenario():
        server = create_server(service=FakeResourceService())
        async with Client(server) as client:
            resources = await client.list_resources()
            assert [str(item.uri) for item in resources.resources] == [
                "literature://guide"
            ]

            templates = await client.list_resource_templates()
            template_uris = {
                item.uri_template for item in templates.resource_templates
            }
            assert template_uris == {
                "literature://papers/{arxiv_id}/metadata",
                "literature://papers/{arxiv_id}/pages/{page_number}",
            }

            guide = await client.read_resource("literature://guide")
            assert isinstance(guide.contents[0], TextResourceContents)
            assert "推荐流程" in guide.contents[0].text

            metadata = await client.read_resource(
                "literature://papers/2601.00001v1/metadata"
            )
            metadata_data = json.loads(metadata.contents[0].text)
            assert metadata_data["paper"]["arxiv_id"] == "2601.00001v1"

            page = await client.read_resource(
                "literature://papers/2601.00001v1/pages/2"
            )
            page_data = json.loads(page.contents[0].text)
            assert page_data["page_number"] == 2

            prompts = await client.list_prompts()
            assert {prompt.name for prompt in prompts.prompts} == {
                "search_literature",
                "analyze_paper",
                "summarize_paper",
            }

            rendered = await client.get_prompt(
                "analyze_paper",
                {
                    "arxiv_id": "2601.00001v1",
                    "question": "主要方法是什么？",
                },
            )
            assert isinstance(rendered.messages[0].content, TextContent)
            assert "literature_retrieve_paper_evidence" in (
                rendered.messages[0].content.text
            )
            assert "主要方法是什么" in rendered.messages[0].content.text

    asyncio.run(scenario())
