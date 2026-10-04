"""Real stdio subprocess smoke test for the model-free client."""

import asyncio

from literature_bot.mcp_integration.client import LiteratureMCPClient


def test_stdio_client_discovers_server_tools():
    async def scenario():
        async with LiteratureMCPClient() as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools}
            assert "literature_search_papers" in names
            assert "literature_retrieve_paper_evidence" in names

            openai_tools = await client.openai_tools()
            assert len(openai_tools) == 4
            assert all(tool["type"] == "function" for tool in openai_tools)

            resources = await client.list_resources()
            assert [str(resource.uri) for resource in resources] == [
                "literature://guide"
            ]
            templates = await client.list_resource_templates()
            assert len(templates) == 2
            assert "推荐流程" in await client.read_resource_text(
                "literature://guide"
            )

            prompts = await client.list_prompts()
            assert {prompt.name for prompt in prompts} == {
                "search_literature",
                "analyze_paper",
                "summarize_paper",
            }
            messages = await client.prompt_messages(
                "search_literature",
                {"topic": "multimodal models", "max_results": "3"},
            )
            assert messages[0]["role"] == "user"
            assert "multimodal models" in messages[0]["content"]

    asyncio.run(scenario())
