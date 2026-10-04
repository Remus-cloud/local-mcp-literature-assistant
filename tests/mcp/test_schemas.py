"""Validation tests for MCP input schemas."""

import pytest
from pydantic import ValidationError

from literature_bot.mcp_integration.schemas import (
    RetrieveEvidenceInput,
    SearchPapersInput,
)


def test_search_input_strips_text_and_limits_result_count():
    params = SearchPapersInput(query="  multimodal models  ", max_results=3)
    assert params.query == "multimodal models"
    assert params.max_results == 3

    with pytest.raises(ValidationError):
        SearchPapersInput(query="valid query", max_results=11)


def test_retrieval_input_rejects_invalid_arxiv_id():
    with pytest.raises(ValidationError):
        RetrieveEvidenceInput(
            arxiv_id="../../secret",
            query="object detection",
        )
