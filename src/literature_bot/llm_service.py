"""OpenAI-compatible client setup and evidence-grounded generation."""

from openai import OpenAI

from .config import Settings
from .models import PaperMetadata, RetrievalHit


def create_llm_client(settings: Settings) -> OpenAI:
    """Create a client for the configured OpenAI-compatible endpoint."""

    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def create_retrieval_query(
    question: str,
    *,
    client: OpenAI,
    model: str,
) -> str:
    """Convert a user question into concise English retrieval keywords."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You convert research questions into concise English "
                    "keywords for searching inside an academic paper. "
                    "Return only the English search query. Do not answer the "
                    "question."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=150,
        extra_body={"enable_thinking": False},
    )
    retrieval_query = (response.choices[0].message.content or "").strip()

    if not retrieval_query:
        raise ValueError("The model did not return a retrieval query.")

    return retrieval_query


def build_evidence_text(retrieval_hits: list[RetrievalHit]) -> str:
    """Format retrieved chunks as page-aware evidence for the model."""

    return "\n\n".join(
        (
            f"[证据 {index}｜论文第 {hit.chunk.page_number} 页｜"
            f"文本块 {hit.chunk.chunk_id}]\n{hit.chunk.text}"
        )
        for index, hit in enumerate(retrieval_hits, start=1)
    )


def answer_question_from_paper(
    question: str,
    paper: PaperMetadata,
    retrieval_hits: list[RetrievalHit],
    *,
    client: OpenAI,
    model: str,
) -> str:
    """Answer strictly from retrieved paper evidence and cite page numbers."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not retrieval_hits:
        raise ValueError("There is no paper evidence available for an answer.")

    prompt = f"""
论文标题：
{paper.title}

arXiv ID：
{paper.arxiv_id}

用户问题：
{question}

论文证据：
{build_evidence_text(retrieval_hits)}

请根据以上论文证据回答用户问题。
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名严谨的中文文献研究助手。"
                    "只能根据用户提供的论文证据回答，不能使用未出现在证据中的事实。"
                    "每个关键结论都应使用类似 `[第 5 页]` 的格式标注来源页码。"
                    "如果证据不足，请明确说明证据不足。"
                    "不要伪造页码、实验数据或结论。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,
        extra_body={"enable_thinking": False},
    )
    answer = (response.choices[0].message.content or "").strip()

    if not answer:
        raise ValueError("The model did not generate an answer.")

    return answer
