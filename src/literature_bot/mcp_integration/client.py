"""A model-free MCP client for the local literature server."""

import os
from pathlib import Path
import sys
from types import TracebackType
from typing import Any, Self

from mcp import Client, StdioServerParameters
from mcp_types import (
    CallToolResult,
    GetPromptResult,
    Prompt,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_server_parameters() -> StdioServerParameters:
    """Describe how to launch the local server using this Python environment."""

    project_root = _project_root()
    source_directory = project_root / "src"
    existing_pythonpath = os.getenv("PYTHONPATH", "")
    pythonpath_parts = [str(source_directory)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "literature_bot.mcp_integration.server"],
        cwd=project_root,
        env={"PYTHONPATH": os.pathsep.join(pythonpath_parts)},
    )


class LiteratureMCPClient:
    """Connect, discover tools, and invoke tools without containing a model."""

    def __init__(
        self,
        server_parameters: StdioServerParameters | None = None,
    ) -> None:
        self.server_parameters = server_parameters or default_server_parameters()
        self._client: Client | None = None

    async def __aenter__(self) -> Self:
        self._client = Client(self.server_parameters)
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc_value, traceback)
            self._client = None

    @property
    def connected_client(self) -> Client:
        if self._client is None:
            raise RuntimeError(
                "The MCP client is not connected. Use 'async with "
                "LiteratureMCPClient() as client'."
            )
        return self._client

    async def list_tools(self) -> list[Tool]:
        result = await self.connected_client.list_tools()
        return result.tools

    async def openai_tools(self) -> list[dict[str, Any]]:
        """Convert discovered MCP tools to OpenAI-compatible tool schemas."""

        tools = await self.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        return await self.connected_client.call_tool(name, arguments or {})

    async def call_tool_data(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a structured tool and raise a readable error on failure."""

        result = await self.call_tool(name, arguments)
        if result.is_error:
            messages = [
                block.text
                for block in result.content
                if getattr(block, "type", None) == "text"
            ]
            raise RuntimeError("\n".join(messages) or f"MCP tool failed: {name}")

        structured = result.structured_content
        if structured is None:
            raise RuntimeError(f"MCP tool returned no structured data: {name}")
        return structured

    async def list_resources(self) -> list[Resource]:
        result = await self.connected_client.list_resources()
        return result.resources

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        result = await self.connected_client.list_resource_templates()
        return result.resource_templates

    async def read_resource(self, uri: str) -> ReadResourceResult:
        return await self.connected_client.read_resource(uri)

    async def read_resource_text(self, uri: str) -> str:
        """Read a text/JSON resource and combine all textual contents."""

        result = await self.read_resource(uri)
        texts = [
            content.text
            for content in result.contents
            if isinstance(content, TextResourceContents)
        ]
        if not texts:
            raise RuntimeError(f"Resource returned no text content: {uri}")
        return "\n\n".join(texts)

    async def list_prompts(self) -> list[Prompt]:
        result = await self.connected_client.list_prompts()
        return result.prompts

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> GetPromptResult:
        return await self.connected_client.get_prompt(name, arguments or {})

    async def prompt_messages(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """Render an MCP prompt into OpenAI-compatible text messages."""

        result = await self.get_prompt(name, arguments)
        messages: list[dict[str, str]] = []
        for message in result.messages:
            if not isinstance(message.content, TextContent):
                raise RuntimeError(
                    f"Prompt '{name}' returned non-text content that this "
                    "client does not yet support."
                )
            role = getattr(message.role, "value", message.role)
            messages.append({"role": str(role), "content": message.content.text})
        return messages
