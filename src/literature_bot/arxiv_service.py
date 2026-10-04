"""Search arXiv and normalize search results."""

import arxiv

from .models import PaperMetadata


ARXIV_USER_AGENT = "literature-bot/0.1 (academic literature search)"
ARXIV_ACCEPT_HEADER = "application/atom+xml, application/xml;q=0.9, */*;q=0.8"


def clean_text(text: str) -> str:
    """Collapse newlines and repeated whitespace into single spaces."""

    return " ".join(text.split())


def create_arxiv_client(
    *,
    page_size: int = 10,
    delay_seconds: float = 3.0,
    num_retries: int = 3,
) -> arxiv.Client:
    """Create the arXiv client using the headers verified in the notebook.

    The installed arxiv package currently exposes the user agent and session as
    private attributes. Keeping that compatibility code here prevents it from
    leaking through the rest of the application.
    """

    arxiv._USER_AGENT = ARXIV_USER_AGENT
    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )
    client._session.headers.update({"Accept": ARXIV_ACCEPT_HEADER})
    return client


def search_arxiv(
    query: str,
    max_results: int = 5,
    *,
    client: arxiv.Client | None = None,
) -> list[PaperMetadata]:
    """Search arXiv and return normalized paper metadata."""

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    active_client = client or create_arxiv_client()
    papers: list[PaperMetadata] = []

    for result in active_client.results(search):
        papers.append(
            PaperMetadata(
                arxiv_id=result.entry_id.rstrip("/").split("/")[-1],
                title=clean_text(result.title),
                authors=[author.name for author in result.authors],
                summary=clean_text(result.summary),
                published=result.published.date().isoformat(),
                updated=result.updated.date().isoformat(),
                entry_url=result.entry_id,
                pdf_url=result.pdf_url,
                categories=list(result.categories),
            )
        )

    return papers


def choose_paper(papers: list[PaperMetadata], selection: int) -> PaperMetadata:
    """Choose a paper using the one-based number displayed to the user."""

    if not papers:
        raise ValueError("There are no papers to choose from.")

    if not 1 <= selection <= len(papers):
        raise ValueError(f"selection must be between 1 and {len(papers)}.")

    return papers[selection - 1]
