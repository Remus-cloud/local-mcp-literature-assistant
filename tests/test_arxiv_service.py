"""Tests for arXiv client setup and result normalization."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from literature_bot.arxiv_service import (
    ARXIV_ACCEPT_HEADER,
    choose_paper,
    clean_text,
    create_arxiv_client,
    search_arxiv,
)


class FakeArxivClient:
    def __init__(self, results):
        self._results = results
        self.received_search = None

    def results(self, search):
        self.received_search = search
        return iter(self._results)


def make_result():
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
    return SimpleNamespace(
        entry_id="https://arxiv.org/abs/2601.00001v1",
        title="  A\nTest   Paper ",
        authors=[SimpleNamespace(name="Ada Researcher")],
        summary="A\n\nshort   summary.",
        published=timestamp,
        updated=timestamp,
        pdf_url="https://arxiv.org/pdf/2601.00001v1",
        categories=["cs.AI"],
    )


def test_create_arxiv_client_uses_verified_accept_header():
    client = create_arxiv_client()

    assert client._session.headers["Accept"] == ARXIV_ACCEPT_HEADER


def test_search_arxiv_normalizes_results_without_network():
    client = FakeArxivClient([make_result()])

    papers = search_arxiv("multimodal models", max_results=1, client=client)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2601.00001v1"
    assert papers[0].title == "A Test Paper"
    assert papers[0].summary == "A short summary."
    assert papers[0].authors == ["Ada Researcher"]


@pytest.mark.parametrize("query", ["", "   "])
def test_search_arxiv_rejects_empty_query(query):
    with pytest.raises(ValueError, match="empty"):
        search_arxiv(query, client=FakeArxivClient([]))


def test_choose_paper_uses_one_based_selection():
    client = FakeArxivClient([make_result(), make_result()])
    papers = search_arxiv("test", max_results=2, client=client)

    assert choose_paper(papers, 1) is papers[0]

    with pytest.raises(ValueError, match="between"):
        choose_paper(papers, 0)


def test_clean_text_collapses_whitespace():
    assert clean_text("one\n\n two\tthree") == "one two three"
