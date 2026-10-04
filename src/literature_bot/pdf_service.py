"""Download arXiv PDFs and extract page-aware text."""

from pathlib import Path
import re
import shutil
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pymupdf

from .arxiv_service import ARXIV_USER_AGENT
from .models import PaperMetadata, PaperPage


def make_safe_filename(value: str) -> str:
    """Convert a value into a Windows-safe filename component."""

    safe_value = re.sub(r'[<>:"/\\|?*]', "_", value).strip(" .")

    if not safe_value:
        raise ValueError("Cannot create a valid filename from this value.")

    return safe_value


def download_paper_pdf(
    paper: PaperMetadata,
    output_directory: str | Path,
    overwrite: bool = False,
) -> Path:
    """Download a paper PDF atomically and validate its file signature."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    filename = f"{make_safe_filename(paper.arxiv_id)}.pdf"
    pdf_path = destination / filename
    temporary_path = destination / f"{filename}.part"

    if pdf_path.exists() and not overwrite:
        return pdf_path

    request = Request(
        paper.pdf_url,
        headers={
            "User-Agent": ARXIV_USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.8",
        },
    )

    try:
        with urlopen(request, timeout=90) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        with temporary_path.open("rb") as pdf_file:
            signature = pdf_file.read(5)

        if signature != b"%PDF-":
            raise ValueError("The downloaded response is not a valid PDF.")

        temporary_path.replace(pdf_path)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        temporary_path.unlink(missing_ok=True)
        raise

    return pdf_path


def clean_page_text(text: str) -> str:
    """Remove blank lines and trailing whitespace from extracted page text."""

    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def extract_pdf_pages(pdf_path: str | Path) -> list[PaperPage]:
    """Extract text from a PDF while preserving one-based page numbers."""

    resolved_path = Path(pdf_path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {resolved_path}")

    pages: list[PaperPage] = []

    with pymupdf.open(resolved_path) as document:
        if document.needs_pass:
            raise ValueError("The PDF is password protected and cannot be read.")

        for page_index, page in enumerate(document):
            text = clean_page_text(page.get_text("text", sort=True))
            pages.append(
                PaperPage(
                    page_number=page_index + 1,
                    text=text,
                    character_count=len(text),
                )
            )

    return pages
