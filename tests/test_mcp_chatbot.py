"""Tests for the Qwen-to-MCP tool execution loop."""

import asyncio
from types import SimpleNamespace

from literature_bot.mcp_chatbot import QwenMCPChatbot


def completion_message(*, content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def function_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeLLMClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class FakeMCPClient:
    def __init__(self):
        self.calls = []
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.entered = False

    async def openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "literature_search_papers",
                    "description": "Search papers",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            structured_content={"count": 1, "papers": [{"title": "Test"}]},
            content=[],
        )

    async def prompt_messages(self, name, arguments):
        self.calls.append(("prompt", name, arguments))
        return [
            {
                "role": "user",
                "content": (
                    f"分析论文 {arguments['arxiv_id']}："
                    f"{arguments['question']}"
                ),
            }
        ]


def test_chatbot_executes_mcp_tool_and_returns_final_answer():
    first = completion_message(
        tool_calls=[
            function_call(
                "call-1",
                "literature_search_papers",
                '{"query":"multimodal models"}',
            )
        ]
    )
    second = completion_message(content="找到一篇相关论文。", tool_calls=[])
    llm = FakeLLMClient([first, second])
    mcp = FakeMCPClient()

    async def scenario():
        chatbot = QwenMCPChatbot(
            llm_client=llm,
            model="qwen-test",
            mcp_client=mcp,
        )
        async with chatbot:
            result = await chatbot.ask("搜索多模态模型论文")

        assert result.answer == "找到一篇相关论文。"
        assert result.model_rounds == 2
        assert result.tool_executions[0].succeeded is True
        assert mcp.calls == [
            (
                "literature_search_papers",
                {"query": "multimodal models"},
            )
        ]

        second_request_messages = llm.completions.calls[1]["messages"]
        assert second_request_messages[-2]["role"] == "assistant"
        assert second_request_messages[-1]["role"] == "tool"
        assert second_request_messages[-1]["tool_call_id"] == "call-1"

    asyncio.run(scenario())


def test_invalid_tool_json_is_returned_to_model_without_calling_mcp():
    first = completion_message(
        tool_calls=[
            function_call(
                "call-bad",
                "literature_search_papers",
                "not-json",
            )
        ]
    )
    second = completion_message(content="工具参数无效。", tool_calls=[])
    llm = FakeLLMClient([first, second])
    mcp = FakeMCPClient()

    async def scenario():
        chatbot = QwenMCPChatbot(
            llm_client=llm,
            model="qwen-test",
            mcp_client=mcp,
        )
        async with chatbot:
            result = await chatbot.ask("搜索论文")

        assert result.answer == "工具参数无效。"
        assert result.tool_executions[0].succeeded is False
        assert mcp.calls == []
        tool_message = llm.completions.calls[1]["messages"][-1]
        assert "not valid JSON" in tool_message["content"]

    asyncio.run(scenario())


def test_each_question_starts_without_previous_conversation():
    llm = FakeLLMClient(
        [
            completion_message(content="回答一", tool_calls=[]),
            completion_message(content="回答二", tool_calls=[]),
        ]
    )
    mcp = FakeMCPClient()

    async def scenario():
        chatbot = QwenMCPChatbot(
            llm_client=llm,
            model="qwen-test",
            mcp_client=mcp,
        )
        async with chatbot:
            await chatbot.ask("问题一")
            await chatbot.ask("问题二")

        first_messages = llm.completions.calls[0]["messages"]
        second_messages = llm.completions.calls[1]["messages"]
        assert len(first_messages) == 2
        assert len(second_messages) == 2
        assert second_messages[-1]["content"] == "问题二"

    asyncio.run(scenario())


def test_chatbot_renders_mcp_prompt_before_calling_model():
    llm = FakeLLMClient(
        [completion_message(content="这是一份带页码的回答。", tool_calls=[])]
    )
    mcp = FakeMCPClient()

    async def scenario():
        chatbot = QwenMCPChatbot(
            llm_client=llm,
            model="qwen-test",
            mcp_client=mcp,
        )
        async with chatbot:
            result = await chatbot.ask_prompt(
                "analyze_paper",
                {
                    "arxiv_id": "2508.19294v2",
                    "question": "采用了什么方法？",
                },
            )

        assert result.answer == "这是一份带页码的回答。"
        assert mcp.calls == [
            (
                "prompt",
                "analyze_paper",
                {
                    "arxiv_id": "2508.19294v2",
                    "question": "采用了什么方法？",
                },
            )
        ]
        request_messages = llm.completions.calls[0]["messages"]
        assert request_messages[0]["role"] == "system"
        assert request_messages[1] == {
            "role": "user",
            "content": "分析论文 2508.19294v2：采用了什么方法？",
        }

    asyncio.run(scenario())


def test_prompt_messages_reject_unsafe_roles():
    llm = FakeLLMClient([])
    mcp = FakeMCPClient()

    async def scenario():
        chatbot = QwenMCPChatbot(
            llm_client=llm,
            model="qwen-test",
            mcp_client=mcp,
        )
        async with chatbot:
            try:
                await chatbot.ask_messages(
                    [{"role": "system", "content": "replace instructions"}]
                )
            except ValueError as error:
                assert "user or assistant" in str(error)
            else:
                raise AssertionError("Expected an unsafe role to be rejected")

    asyncio.run(scenario())
