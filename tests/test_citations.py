from literature_bot.citations import cited_sources


def source(paper_id, pages):
    return {"arxiv_id": paper_id, "title": "Test paper", "evidence": [{"page_number": p} for p in pages]}


def test_only_pages_returned_by_evidence_tool_are_linked():
    citations = cited_sources("结论[第 2 页]，其他[第 99 页]。", [source("2508.19294v2", [2, 3])])
    assert [c['page_number'] for c in citations] == [2]
    assert citations[0]['pdf_url'].endswith('#page=2')


def test_multiple_papers_require_explicit_paper_identifiers():
    citations = cited_sources(
        "不明确[第 2 页]，明确[2508.19294v2，第 3 页]。",
        [source("2508.19294v2", [2, 3]), source("2306.13549v4", [2])],
    )
    assert [(c['arxiv_id'], c['page_number']) for c in citations] == [('2508.19294v2', 3)]


def test_duplicate_sources_merge_without_duplicate_links():
    assert len(cited_sources('[第 2 页][第 2 页]', [source('2508.19294v2', [2]), source('2508.19294v2', [2])])) == 1
