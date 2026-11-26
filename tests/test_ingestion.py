from pathlib import Path
import fitz
import pytest

from core.exceptions import DocumentIngestionError
from ingestion.chunker import chunk_parsed_document
from ingestion.hashing import compute_sha256, validate_pdf_file
from ingestion.pdf_extractor import parse_civic_pdf
from ingestion.table_formatter import rows_to_markdown


def _create_sample_pdf(pdf_path: Path, text: str = "PM-KISAN provides Rs. 6,000 annually.") -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_rows_to_markdown_formats_and_escapes_pipes() -> None:
    rows = [
        ["Category", "Income Limit | Annual"],
        ["EWS", "Below Rs. 8,00,000"],
        [None, None],
    ]
    md = rows_to_markdown(rows)
    assert "| Category | Income Limit \\| Annual |" in md
    assert "| --- | --- |" in md
    assert "| EWS | Below Rs. 8,00,000 |" in md


def test_pdf_validation_rejects_non_pdf_and_empty_files(tmp_path: Path) -> None:
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(DocumentIngestionError, match="Only PDF documents"):
        validate_pdf_file(txt_file)

    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(DocumentIngestionError, match="empty"):
        validate_pdf_file(empty_pdf)


def test_parse_and_chunk_civic_pdf(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(
        tmp_path / "welfare_scheme.pdf",
        text="Eligibility criteria: Small and marginal farmers holding cultivable land.",
    )
    digest = compute_sha256(pdf_path)
    assert len(digest) == 64

    parsed = parse_civic_pdf(pdf_path)
    assert parsed.filename == "welfare_scheme.pdf"
    assert parsed.page_count == 1
    assert parsed.file_hash == digest

    chunks = chunk_parsed_document(parsed, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "welfare_scheme.pdf"
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["content_type"] == "prose"