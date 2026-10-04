"""Tests for PDF filenames, downloads, and page extraction."""

from io import BytesIO

import pymupdf
import pytest

from literature_bot.models import PaperMetadata
from literature_bot import pdf_service


def make_paper() -> PaperMetadata:
    return PaperMetadata(
        arxiv_id="2601.00001v1",
        title="Test Paper",
        authors=["Ada Researcher"],
        summary="Summary",
        published="2026-01-02",
        updated="2026-01-02",
        entry_url="https://arxiv.org/abs/2601.00001v1",
        pdf_url="https://arxiv.org/pdf/2601.00001v1",
        categories=["cs.AI"],
    )


def make_pdf_bytes(text: str = "Hello literature bot") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class FakeResponse(BytesIO):
    def __init__(self, content: bytes):
        super().__init__(content)
        self.headers = {"Content-Type": "application/pdf"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def test_make_safe_filename_replaces_windows_reserved_characters():
    assert pdf_service.make_safe_filename('a<b>:c/"d"') == "a_b__c__d_"


def test_download_paper_pdf_writes_valid_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pdf_service,
        "urlopen",
        lambda request, timeout: FakeResponse(make_pdf_bytes()),
    )

    pdf_path = pdf_service.download_paper_pdf(make_paper(), tmp_path)

    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    assert not (tmp_path / "2601.00001v1.pdf.part").exists()


def test_download_rejects_non_pdf_and_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pdf_service,
        "urlopen",
        lambda request, timeout: FakeResponse(b"<html>error</html>"),
    )

    with pytest.raises(ValueError, match="not a valid PDF"):
        pdf_service.download_paper_pdf(make_paper(), tmp_path)

    assert not (tmp_path / "2601.00001v1.pdf.part").exists()


def test_extract_pdf_pages_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(make_pdf_bytes("Evidence on page one"))

    pages = pdf_service.extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Evidence on page one" in pages[0].text
    assert pages[0].character_count == len(pages[0].text)
