from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_settings
from core.logger import get_logger
from ingestion.models import ParsedDocument

logger = get_logger(__name__)


def chunk_parsed_document(
    parsed_doc: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """Split extracted PDF prose and tables into citation-anchored LangChain Documents."""
    settings = get_settings()
    resolved_size = chunk_size or settings.chunk_size
    resolved_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_size,
        chunk_overlap=resolved_overlap,
        separators=["\n\n", "\n", ". ", "। ", " ", ""],
    )

    chunks: List[Document] = []
    chunk_idx = 0

    for page in parsed_doc.pages:
        if page.raw_text.strip():
            prose_splits = splitter.split_text(page.raw_text.strip())
            for split_text in prose_splits:
                cleaned = split_text.strip()
                if not cleaned:
                    continue
                chunks.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "source": parsed_doc.filename,
                            "file_hash": parsed_doc.file_hash,
                            "page": page.page_number,
                            "chunk_id": f"{parsed_doc.filename}:p{page.page_number}:c{chunk_idx}",
                            "content_type": "prose",
                        },
                    )
                )
                chunk_idx += 1

        for table_idx, md_table in enumerate(page.markdown_tables, start=1):
            table_block = (
                f"[Structured Table {table_idx} — {parsed_doc.filename}, Page {page.page_number}]\n"
                f"{md_table}"
            )
            chunks.append(
                Document(
                    page_content=table_block,
                    metadata={
                        "source": parsed_doc.filename,
                        "file_hash": parsed_doc.file_hash,
                        "page": page.page_number,
                        "chunk_id": f"{parsed_doc.filename}:p{page.page_number}:t{table_idx}",
                        "content_type": "table",
                    },
                )
            )
            chunk_idx += 1

    logger.info(
        "Chunked '%s' into %d citation-tagged segments.",
        parsed_doc.filename,
        len(chunks),
    )
    return chunks