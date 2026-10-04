"""Streamlit interface for the stateless Qwen + literature MCP chatbot."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from literature_bot.config import load_settings
from literature_bot.citations import cited_sources
from literature_bot.mcp_chatbot import ChatbotResult, QwenMCPChatbot


WELCOME_TEXT = """
我可以根据研究主题搜索 arXiv 文献、查看论文信息、读取 PDF 正文，
并根据正文证据回答问题。涉及论文内容的关键结论会尽量使用
`[第 X 页]` 标注来源。

你可以选择快捷任务，也可以直接在页面底部输入问题。
""".strip()


@dataclass(frozen=True)
class Submission:
    """One UI request waiting to be sent to the chatbot."""

    display_text: str
    question: str | None = None
    prompt_name: str | None = None
    prompt_arguments: dict[str, str] = field(default_factory=dict)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def _answer_submission(submission: Submission) -> ChatbotResult:
    settings = load_settings(_project_root() / ".env")
    chatbot = QwenMCPChatbot.from_settings(settings)

    async with chatbot:
        if submission.prompt_name is not None:
            result = await chatbot.ask_prompt(
                submission.prompt_name,
                submission.prompt_arguments,
            )
        else:
            if submission.question is None:
                raise ValueError("A submission must contain a question or prompt.")
            result = await chatbot.ask(submission.question)
        pages = []
        for citation in cited_sources(result.answer, result.evidence_sources):
            try:
                paper_id = citation['arxiv_id']
                page_number = citation['page_number']
                raw = await chatbot.mcp_client.read_resource_text(
                    f"literature://papers/{paper_id}/pages/{page_number}"
                )
                resource = json.loads(raw)
                if resource.get('arxiv_id') != paper_id or resource.get('page_number') != page_number:
                    raise ValueError("页面来源与引用不一致。")
                citation['text'] = resource['text']
            except Exception:
                citation['error'] = "原文暂时无法读取，请通过 PDF 链接核对。"
            pages.append(citation)
        return replace(result, citation_pages=pages)


def answer_submission(submission: Submission) -> ChatbotResult:
    """Run one isolated async chatbot request from Streamlit's sync script."""

    return asyncio.run(_answer_submission(submission))


def format_error(error: BaseException) -> str:
    """Expose actionable leaf errors instead of TaskGroup wrapper messages."""

    if isinstance(error, BaseExceptionGroup):
        messages: list[str] = []
        for child in error.exceptions:
            message = format_error(child)
            if message and message not in messages:
                messages.append(message)
        return "；".join(messages) or str(error)
    return str(error).strip() or type(error).__name__


def _initialize_state() -> None:
    if "active_prompt" not in st.session_state:
        st.session_state.active_prompt = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _activate_prompt(prompt_name: str) -> None:
    st.session_state.active_prompt = prompt_name


def _render_quick_actions() -> Submission | None:
    st.subheader("快捷任务")
    search_column, analyze_column, summarize_column = st.columns(3)

    search_column.button(
        "🔎 搜索文献",
        use_container_width=True,
        on_click=_activate_prompt,
        args=("search_literature",),
    )
    analyze_column.button(
        "📄 分析论文",
        use_container_width=True,
        on_click=_activate_prompt,
        args=("analyze_paper",),
    )
    summarize_column.button(
        "📝 总结论文",
        use_container_width=True,
        on_click=_activate_prompt,
        args=("summarize_paper",),
    )

    active_prompt = st.session_state.active_prompt
    if active_prompt is None:
        return None

    if st.button("关闭快捷任务", type="tertiary"):
        st.session_state.active_prompt = None
        st.rerun()

    if active_prompt == "search_literature":
        with st.form("search_literature_form"):
            st.markdown("#### 搜索 arXiv 文献")
            topic = st.text_input(
                "研究主题",
                placeholder="例如：multimodal large language models",
            )
            max_results = st.slider("返回论文数量", 1, 10, 5)
            submitted = st.form_submit_button(
                "开始搜索",
                use_container_width=True,
            )
        if submitted:
            if not topic.strip():
                st.warning("请输入研究主题。")
                return None
            return Submission(
                display_text=f"搜索文献：{topic.strip()}（{max_results} 篇）",
                prompt_name="search_literature",
                prompt_arguments={
                    "topic": topic.strip(),
                    "max_results": str(max_results),
                },
            )

    if active_prompt == "analyze_paper":
        with st.form("analyze_paper_form"):
            st.markdown("#### 根据正文证据分析论文")
            arxiv_id = st.text_input(
                "arXiv ID",
                placeholder="例如：2508.19294v2",
            )
            question = st.text_area(
                "论文问题",
                placeholder="例如：这篇论文采用了什么方法？",
            )
            submitted = st.form_submit_button(
                "分析论文",
                use_container_width=True,
            )
        if submitted:
            if not arxiv_id.strip() or not question.strip():
                st.warning("请输入 arXiv ID 和论文问题。")
                return None
            return Submission(
                display_text=(
                    f"分析论文 {arxiv_id.strip()}：{question.strip()}"
                ),
                prompt_name="analyze_paper",
                prompt_arguments={
                    "arxiv_id": arxiv_id.strip(),
                    "question": question.strip(),
                },
            )

    if active_prompt == "summarize_paper":
        with st.form("summarize_paper_form"):
            st.markdown("#### 总结论文")
            arxiv_id = st.text_input(
                "arXiv ID",
                placeholder="例如：2508.19294v2",
            )
            focus = st.text_input(
                "总结重点",
                value="研究问题、方法、实验与结论",
            )
            submitted = st.form_submit_button(
                "生成总结",
                use_container_width=True,
            )
        if submitted:
            if not arxiv_id.strip() or not focus.strip():
                st.warning("请输入 arXiv ID 和总结重点。")
                return None
            return Submission(
                display_text=(
                    f"总结论文 {arxiv_id.strip()}，重点：{focus.strip()}"
                ),
                prompt_name="summarize_paper",
                prompt_arguments={
                    "arxiv_id": arxiv_id.strip(),
                    "focus": focus.strip(),
                },
            )

    return None


def _render_citations(pages: list[dict[str, Any]]) -> None:
    if not pages:
        return
    st.caption("引用原文（页码为 PDF 文件页序）")
    for page in pages:
        with st.expander(
            f"查看原文 · 第 {page['page_number']} 页 · {page['title']}"
        ):
            st.markdown(f"[在 PDF 中打开第 {page['page_number']} 页]({page['pdf_url']})")
            st.caption(f"arXiv：{page['arxiv_id']} · PDF 第 {page['page_number']} 页")
            if page.get('error'):
                st.warning(page['error'])
            else:
                st.text(page.get('text') or '该页没有可提取的文字。')


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_citations(message.get("citation_pages", []))
        tool_names = message.get("tool_names", [])
        if tool_names:
            st.caption("调用 MCP Tools：" + " → ".join(tool_names))


def _append_result(submission: Submission) -> None:
    user_message = {"role": "user", "content": submission.display_text}
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在连接 Qwen 和 Literature MCP Server…"):
                result = answer_submission(submission)
        except Exception as error:
            error_text = f"处理失败：{format_error(error)}"
            st.error(error_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_text}
            )
            return

        st.markdown(result.answer)
        _render_citations(result.citation_pages)
        tool_names = [execution.name for execution in result.tool_executions]
        if tool_names:
            st.caption("调用 MCP Tools：" + " → ".join(tool_names))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "tool_names": tool_names,
                "citation_pages": result.citation_pages,
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="文献搜索与问答机器人",
        page_icon="📚",
        layout="centered",
    )
    _initialize_state()

    st.title("📚 文献搜索与问答机器人")
    st.markdown(WELCOME_TEXT)

    with st.sidebar:
        st.subheader("运行说明")
        st.markdown(
            "当前版本不保存长期对话记录。页面中的消息只用于本次浏览器会话显示。"
        )
        if st.button("清空页面消息", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    quick_submission = _render_quick_actions()
    st.divider()

    for message in st.session_state.messages:
        _render_message(message)

    question = st.chat_input(
        "直接提问，例如：帮我搜索具身智能相关论文",
        max_chars=2_000,
    )
    submission = quick_submission
    if question:
        submission = Submission(
            display_text=question.strip(),
            question=question.strip(),
        )

    if submission is not None:
        _append_result(submission)


if __name__ == "__main__":
    main()
