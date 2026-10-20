"""Connect Qwen tool calling to the model-free literature MCP client."""

import argparse
import asyncio
from dataclasses import dataclass, field
from functools import partial
import json
import os
from pathlib import Path
import sys
from types import TracebackType
from typing import Any, Self

import anyio
from openai import OpenAI

from .config import Settings, load_settings
from .llm_service import create_llm_client
from .mcp_integration.client import LiteratureMCPClient
from .evidence_binding import BINDING_PROMPT, EvidenceRegistry
from .provider import request_options


DEFAULT_SYSTEM_PROMPT = """
你是一名严谨的中文文献研究助手，可以使用 MCP 文献工具。

工作规则：
1. 当用户要求搜索或推荐文献时，先调用 literature_search_papers。
2. 当用户询问一篇论文的正文内容、方法、实验或结论时，必须先调用
   literature_retrieve_paper_evidence，并将中文问题转换为简洁的英文检索词。
3. 只能根据工具返回的论文信息和证据回答，不得编造论文、页码、数据或结论。
4. 引用正文证据时使用 `[第 X 页]` 格式；证据不足时明确说明。
5. 不需要读取正文时，不要下载或准备 PDF。
6. 默认使用中文回答，保留论文英文标题和专业术语。
7. 同时引用多篇论文时，使用 `[arXiv ID，第 X 页]`，明确每个页码所属论文。
""".strip()


@dataclass(frozen=True)
class ToolExecution:
    """One MCP tool call performed during a single user request."""

    name: str
    arguments: dict[str, Any]
    succeeded: bool


@dataclass(frozen=True)
class ChatbotResult:
    """Final answer plus an inspectable, non-persistent tool trace."""

    answer: str
    tool_executions: list[ToolExecution]
    model_rounds: int
    evidence_sources: list[dict[str, Any]] = field(default_factory=list)
    citation_pages: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    binding: dict[str, Any] = field(default_factory=dict)
    search_runs: list[dict[str, Any]] = field(default_factory=list)


class QwenMCPChatbot:
    """A stateless Qwen chatbot that can call tools through MCP.

    The MCP connection remains open while the object is inside ``async with``,
    but every call to :meth:`ask` starts a fresh message list. Consequently,
    paper preparation can be reused in one process while conversation history
    is never persisted between questions.
    """

    def __init__(
        self,
        *,
        llm_client: OpenAI,
        model: str,
        mcp_client: LiteratureMCPClient | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_rounds: int = 8,
        verifier: Any = None,
        verification_mode: str | None = None,
        binding_mode: str | None = None,
        numeric_mode: str | None = None,
        diagnostics_mode: str | None = None,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")

        self.llm_client = llm_client
        self.model = model
        self.mcp_client = mcp_client or LiteratureMCPClient()
        self.system_prompt = system_prompt.strip()
        self.max_tool_rounds = max_tool_rounds
        self._tools: list[dict[str, Any]] | None = None
        self.binding_mode = binding_mode or os.getenv("LITERATURE_BOT_BINDING", "off")
        if self.binding_mode not in {"off", "structured"}:
            raise ValueError("LITERATURE_BOT_BINDING must be off or structured")
        self.binding_format = os.getenv("LITERATURE_BOT_BINDING_FORMAT", "json_object")
        if self.binding_format not in {"json_object", "prompt_only"}:
            raise ValueError("LITERATURE_BOT_BINDING_FORMAT must be json_object or prompt_only")
        self.numeric_mode = numeric_mode if numeric_mode is not None else os.getenv("LITERATURE_BOT_NUMERIC_VERIFICATION", "off")
        if self.numeric_mode not in {"off", "shadow"}:
            raise ValueError("LITERATURE_BOT_NUMERIC_VERIFICATION must be off or shadow")
        self.diagnostics_mode = diagnostics_mode if diagnostics_mode is not None else os.getenv("LITERATURE_BOT_DIAGNOSTICS", "off")
        if self.diagnostics_mode not in {"off", "shadow"}:
            raise ValueError("LITERATURE_BOT_DIAGNOSTICS must be off or shadow")
        mode = verification_mode or os.getenv("LITERATURE_BOT_VERIFICATION", "off")
        if mode not in {"off", "shadow"}:
            raise ValueError("LITERATURE_BOT_VERIFICATION must be off or shadow")
        self.verifier = verifier
        if self.verifier is None and mode == "shadow":
            from .verification import AnswerVerifier
            from .verification.claims import QwenClaimExtractor

            self.verifier = AnswerVerifier(QwenClaimExtractor(llm_client, model))

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        **kwargs: Any,
    ) -> "QwenMCPChatbot":
        return cls(
            llm_client=create_llm_client(settings),
            model=settings.model,
            **kwargs,
        )

    async def __aenter__(self) -> Self:
        await self.mcp_client.__aenter__()
        self._tools = await self.mcp_client.openai_tools()
        if not self._tools:
            await self.mcp_client.__aexit__(None, None, None)
            raise RuntimeError("The MCP server exposed no tools.")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._tools = None
        await self.mcp_client.__aexit__(exc_type, exc_value, traceback)

    async def ask(self, question: str) -> ChatbotResult:
        """Answer one independent user question, including any MCP tool loop."""

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if self._tools is None:
            raise RuntimeError(
                "The chatbot is not connected. Use 'async with "
                "QwenMCPChatbot(...) as chatbot'."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        return await self._run_tool_loop(messages)

    async def ask_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
    ) -> ChatbotResult:
        """Render one MCP prompt and answer it through the normal tool loop."""

        prompt_name = prompt_name.strip()
        if not prompt_name:
            raise ValueError("Prompt name cannot be empty.")
        prompt_messages = await self.mcp_client.prompt_messages(
            prompt_name,
            arguments or {},
        )
        return await self.ask_messages(prompt_messages)

    async def ask_messages(
        self,
        prompt_messages: list[dict[str, str]],
    ) -> ChatbotResult:
        """Answer validated user/assistant messages rendered by MCP prompts."""

        if self._tools is None:
            raise RuntimeError(
                "The chatbot is not connected. Use 'async with "
                "QwenMCPChatbot(...) as chatbot'."
            )
        if not prompt_messages:
            raise ValueError("Prompt messages cannot be empty.")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for message in prompt_messages:
            role = message.get("role", "").strip()
            content = message.get("content", "").strip()
            if role not in {"user", "assistant"}:
                raise ValueError(
                    "Prompt messages may only use user or assistant roles."
                )
            if not content:
                raise ValueError("Prompt message content cannot be empty.")
            messages.append({"role": role, "content": content})

        return await self._run_tool_loop(messages)

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
    ) -> ChatbotResult:
        """Run Qwen and resolve requested MCP tools until text is returned."""

        if self._tools is None:
            raise RuntimeError(
                "The chatbot is not connected. Use 'async with "
                "QwenMCPChatbot(...) as chatbot'."
            )

        allowed_tool_names = {
            tool["function"]["name"]
            for tool in self._tools
        }
        executions: list[ToolExecution] = []
        evidence_sources: list[dict[str, Any]] = []
        search_runs: list[dict[str, Any]] = []
        registry = EvidenceRegistry()
        if self.binding_mode == "structured":
            # Keep ordinary search-only requests and MCP protocol unchanged.
            messages = [dict(m) for m in messages]
            messages[0]["content"] += "\n\n" + BINDING_PROMPT

        for round_number in range(1, self.max_tool_rounds + 1):
            response = await anyio.to_thread.run_sync(
                partial(
                    self.llm_client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    tools=self._tools,
                    tool_choice="auto",
                    max_tokens=3_500 if self.binding_mode == "structured" and registry.entries else 2_000,
                    **request_options(self.llm_client),
                    **({"response_format": {"type": "json_object"}}
                       if self.binding_mode == "structured" and registry.entries
                       and self.binding_format == "json_object" else {}),
                )
            )
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                answer = (assistant_message.content or "").strip()
                if not answer:
                    raise RuntimeError("Qwen returned neither text nor tool calls.")
                binding = {"status": "disabled" if self.binding_mode == "off" else "no_body_evidence"}
                if self.binding_mode == "structured" and registry.entries:
                    try:
                        if getattr(response.choices[0], "finish_reason", None) == "length":
                            raise ValueError("truncated_binding_response")
                        answer, binding = registry.render(answer)
                    except (ValueError, TypeError) as error:
                        binding = {
                            "status": "invalid_schema", "error_type": type(error).__name__,
                            "raw_response": answer, "claims": [],
                        }
                        answer = "【本次结构化证据绑定失败，以下为未验证的原始输出】\n\n" + answer
                verification = {"mode": "off", "status": "disabled"}
                if self.verifier is not None:
                    try:
                        if binding["status"] == "invalid_schema":
                            verification = {"mode": "shadow", "status": "skipped_binding_failure"}
                        elif binding["status"] == "structured":
                            verification = await anyio.to_thread.run_sync(
                                partial(self.verifier.verify_bound, answer, evidence_sources, binding)
                            )
                        else:
                            verification = await anyio.to_thread.run_sync(
                                partial(self.verifier.verify, answer, evidence_sources)
                            )
                    except Exception as error:
                        # Diagnostics must never discard a successfully generated
                        # answer or expose raw provider exception text/secrets.
                        verification = {
                            "mode": "shadow", "status": "verification_error",
                            "error_type": type(error).__name__, "answer_modified": False,
                        }
                if self.numeric_mode == "shadow":
                    from .verification.numeric import audit_numeric
                    # Independent of MiniCheck and its failure/latency. No model
                    # request, no answer rewriting, no change to semantic scores.
                    question = "\n".join(str(m.get("content", "")) for m in messages
                                         if m.get("role") == "user")
                    verification["numeric"] = audit_numeric(
                        answer, evidence_sources, binding, question,
                    )
                if self.diagnostics_mode == "shadow":
                    from .verification.diagnostics import build_diagnostics
                    # Post-generation only: never feed these candidate labels
                    # back into the model or overwrite any upstream diagnostic.
                    verification["unified"] = build_diagnostics(
                        answer, evidence_sources, binding, verification,
                        numeric_report=verification.get("numeric", {"status": "disabled"}),
                    )
                return ChatbotResult(
                    answer=answer,
                    tool_executions=executions,
                    model_rounds=round_number,
                    evidence_sources=evidence_sources,
                    verification=verification,
                    binding=binding,
                    search_runs=search_runs,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments, argument_error = self._parse_arguments(
                    tool_call.function.arguments
                )

                if tool_name not in allowed_tool_names:
                    tool_result = {
                        "ok": False,
                        "error": f"Unknown MCP tool requested: {tool_name}",
                    }
                    succeeded = False
                elif argument_error is not None:
                    tool_result = {"ok": False, "error": argument_error}
                    succeeded = False
                else:
                    mcp_result = await self.mcp_client.call_tool(
                        tool_name,
                        arguments,
                    )
                    tool_result = self._serialize_mcp_result(mcp_result)
                    succeeded = not bool(mcp_result.is_error)
                    if succeeded and tool_name == "literature_search_papers":
                        data = tool_result.get("data")
                        if isinstance(data, dict):
                            search_runs.append({
                                key: data.get(key) for key in
                                ("query", "count", "candidate_count", "retrieval_mode", "reranking_applied")
                            })
                    if succeeded and tool_name == "literature_retrieve_paper_evidence":
                        data = tool_result.get("data")
                        if isinstance(data, dict):
                            if self.binding_mode == "structured":
                                data = registry.register(data)
                                tool_result["data"] = data
                                # Do not send a second unannotated duplicate to the model.
                                tool_result["messages"] = []
                            evidence_sources.append(data)

                executions.append(
                    ToolExecution(
                        name=tool_name,
                        arguments=arguments,
                        succeeded=succeeded,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            tool_result,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )

        raise RuntimeError(
            f"Qwen exceeded the maximum of {self.max_tool_rounds} tool rounds."
        )

    @staticmethod
    def _parse_arguments(raw_arguments: str) -> tuple[dict[str, Any], str | None]:
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            return {}, f"Tool arguments were not valid JSON: {error.msg}"

        if not isinstance(parsed, dict):
            return {}, "Tool arguments must be a JSON object."
        return parsed, None

    @staticmethod
    def _serialize_mcp_result(result: Any) -> dict[str, Any]:
        structured = getattr(result, "structured_content", None)
        text_blocks = [
            block.text
            for block in getattr(result, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        return {
            "ok": not bool(getattr(result, "is_error", False)),
            "data": structured,
            "messages": text_blocks,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def _run_cli(question: str | None) -> None:
    settings = load_settings(_project_root() / ".env")
    chatbot = QwenMCPChatbot.from_settings(settings)

    async with chatbot:
        if question is not None:
            result = await chatbot.ask(question)
            print(result.answer)
            return

        print("Literature MCP Chatbot 已启动。输入 quit 退出。")
        while True:
            try:
                user_input = (await asyncio.to_thread(input, "\n问题：")).strip()
            except EOFError:
                break
            if user_input.lower() in {"quit", "exit"}:
                break

            try:
                result = await chatbot.ask(user_input)
            except Exception as error:
                print(f"\n处理失败：{error}")
                continue

            if result.tool_executions:
                called = ", ".join(
                    execution.name for execution in result.tool_executions
                )
                print(f"\n调用工具：{called}")
            print("\n回答：")
            print(result.answer)


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run the stateless Qwen literature chatbot over local MCP."
    )
    parser.add_argument(
        "--question",
        help="Ask one question and exit; omit for an interactive loop.",
    )
    args = parser.parse_args()
    asyncio.run(_run_cli(args.question))


if __name__ == "__main__":
    main()
