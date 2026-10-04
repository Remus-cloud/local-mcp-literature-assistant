"""Match answer citations to actual tool evidence before exposing pages."""

import re
from typing import Any
from urllib.parse import quote


def cited_sources(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only link supported pages; bare page citations require a single paper."""

    papers: dict[str, dict[str, Any]] = {}
    for source in sources:
        paper_id = source.get("arxiv_id")
        if not isinstance(paper_id, str) or not paper_id:
            continue
        paper = papers.setdefault(
            paper_id, {"arxiv_id": paper_id, "title": source.get("title", paper_id), "pages": set()}
        )
        for evidence in source.get("evidence", []):
            page = evidence.get("page_number")
            if isinstance(page, int) and not isinstance(page, bool) and page > 0:
                paper["pages"].add(page)

    bare_pages = {int(page) for page in re.findall(r"\[第\s*(\d+)\s*页\]", answer)}
    result = []
    for paper_id, paper in papers.items():
        explicit = re.findall(
            r"\[" + re.escape(paper_id) + r"\s*[,，]\s*第\s*(\d+)\s*页\]", answer
        )
        pages = {int(page) for page in explicit}
        if len(papers) == 1:
            pages |= bare_pages
        for page in sorted(pages & paper["pages"]):
            result.append({
                "arxiv_id": paper_id,
                "title": paper["title"],
                "page_number": page,
                "pdf_url": f"https://arxiv.org/pdf/{quote(paper_id, safe='/')}#page={page}",
            })
    return result
