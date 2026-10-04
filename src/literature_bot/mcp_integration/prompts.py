"""Register user-controlled prompt templates for literature workflows."""

from mcp.server.mcpserver import MCPServer


def _required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def register_prompts(server: MCPServer) -> None:
    """Attach reusable search, analysis, and summary prompt templates."""

    @server.prompt(
        name="search_literature",
        title="Search Academic Literature",
        description=(
            "Search arXiv for a topic and present a concise list of relevant "
            "papers."
        ),
    )
    def search_literature(topic: str, max_results: str = "5") -> str:
        """Create a user message for an arXiv literature search."""

        topic = _required(topic, "topic")
        try:
            count = int(max_results)
        except ValueError as error:
            raise ValueError("max_results must be an integer from 1 to 10") from error
        if not 1 <= count <= 10:
            raise ValueError("max_results must be between 1 and 10")

        return (
            f"请使用 literature_search_papers 搜索与“{topic}”相关的 arXiv "
            f"论文，最多返回 {count} 篇。列出每篇论文的英文标题、作者、"
            "arXiv ID，并简要说明它与主题的关系。不要下载 PDF。"
        )

    @server.prompt(
        name="analyze_paper",
        title="Analyze a Paper with Evidence",
        description=(
            "Answer a specific question about one paper using retrieved, "
            "page-cited evidence."
        ),
    )
    def analyze_paper(arxiv_id: str, question: str) -> str:
        """Create a user message for evidence-grounded paper analysis."""

        arxiv_id = _required(arxiv_id, "arxiv_id")
        question = _required(question, "question")
        return (
            f"请分析 arXiv 论文 `{arxiv_id}` 并回答以下问题：\n\n"
            f"{question}\n\n"
            "必须调用 literature_retrieve_paper_evidence 检索论文正文。"
            "将问题转换为简洁的英文检索词，并且只根据返回证据回答。"
            "每个关键结论使用 `[第 X 页]` 标注来源；证据不足时明确说明。"
        )

    @server.prompt(
        name="summarize_paper",
        title="Summarize a Paper with Citations",
        description=(
            "Summarize one paper's problem, methods, experiments, and conclusions "
            "with page citations."
        ),
    )
    def summarize_paper(
        arxiv_id: str,
        focus: str = "研究问题、方法、实验与结论",
    ) -> str:
        """Create a user message for a page-cited paper summary."""

        arxiv_id = _required(arxiv_id, "arxiv_id")
        focus = _required(focus, "focus")
        return (
            f"请总结 arXiv 论文 `{arxiv_id}`，重点关注：{focus}。\n\n"
            "必须调用 literature_retrieve_paper_evidence 获取论文证据。"
            "总结应结构清晰，只使用返回的论文内容，并为关键结论标注"
            " `[第 X 页]`。如果检索证据不足，请指出缺失部分。"
        )
