from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
import pdfplumber

from core.exceptions import DocumentIngestionError
from core.logger import get_logger
from ingestion.hashing import compute_sha256, validate_pdf_file
from ingestion.models import ExtractedPage, ParsedDocument
from ingestion.table_formatter import rows_to_markdown

logger = get_logger(__name__)


def _extract_tables_by_page(pdf_path: Path) -> Dict[int, List[str]]:
    """Extract structured tables per 1-indexed page using pdfplumber."""
    tables_by_page: Dict[int, List[str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables() or []
            md_tables: List[str] = []
            for table in raw_tables:
                md = rows_to_markdown(table)
                if md:
                    md_tables.append(md)
            if md_tables:
                tables_by_page[page_idx] = md_tables
    return tables_by_page


def parse_civic_pdf(file_path: Path, max_size_mb: int = 50) -> ParsedDocument:
    """Extract page-level text and Markdown tables from a civic scheme PDF."""
    path = Path(file_path)
    validate_pdf_file(path, max_size_mb=max_size_mb)
    file_hash = compute_sha256(path)

    try:
        tables_by_page = _extract_tables_by_page(path)
        pages: List[ExtractedPage] = []
        total_tables = 0

        with fitz.open(path) as doc:
            page_count = len(doc)
            if page_count == 0:
                raise DocumentIngestionError(f"PDF '{path.name}' contains zero pages.")

            raw_meta = doc.metadata or {}
            metadata = {k: v for k, v in raw_meta.items() if v}

            for idx in range(page_count):
                page_num = idx + 1
                page_text = doc[idx].get_text("text") or ""
                page_tables = tables_by_page.get(page_num, [])
                total_tables += len(page_tables)

                pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        raw_text=page_text,
                        markdown_tables=page_tables,
                    )
                )

        if not any(p.combined_content.strip() for p in pages):
            raise DocumentIngestionError(
                f"No extractable text or tables found in '{path.name}' (may be a scanned image PDF)."
            )

        logger.info(
            "Parsed '%s': %d pages, %d tables extracted.",
            path.name,
            page_count,
            total_tables,
        )
        return ParsedDocument(
            filename=path.name,
            file_hash=file_hash,
            page_count=page_count,
            table_count=total_tables,
            pages=pages,
            metadata=metadata,
        )
    except DocumentIngestionError:
        raise
    except Exception as exc:
        raise DocumentIngestionError(
            f"Failed to parse PDF '{path.name}': {exc}"
        ) from exc