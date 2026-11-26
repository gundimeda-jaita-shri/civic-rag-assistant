import fitz  # PyMuPDF
import pdfplumber
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a 2D list of table cells into clean Markdown table rows."""
    cleaned_rows = []
    for row in table:
        if not row or not any(cell and cell.strip() for cell in row):
            continue
        cleaned_cells = [
            (cell.replace("\n", " ").strip() if cell else "") for cell in row
        ]
        cleaned_rows.append("| " + " | ".join(cleaned_cells) + " |")

    if len(cleaned_rows) < 2:
        return "\n".join(cleaned_rows)

    col_count = len(cleaned_rows[0].split("|")) - 2
    separator = "| " + " | ".join(["---"] * max(col_count, 1)) + " |"
    return "\n".join([cleaned_rows[0], separator] + cleaned_rows[1:])


def parse_pdf(file_path: str, source_name: str | None = None) -> List[Document]:
    """
    Parse a PDF into page-aware LangChain Documents:
    1. Extracts structured tables per page as Markdown via pdfplumber.
    2. Extracts narrative text blocks with bounding-box metadata via PyMuPDF.
    """
    doc_name = source_name or file_path.split("/")[-1].split("\\")[-1]
    raw_documents: List[Document] = []

    # 1. Extract structured tables via pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables):
                md_table = _table_to_markdown(table)
                if md_table.strip():
                    raw_documents.append(
                        Document(
                            page_content=f"[Table {t_idx + 1} on Page {page_num}]\n{md_table}",
                            metadata={
                                "source": doc_name,
                                "page": page_num,
                                "chunk_type": "table",
                                "bbox": None,
                            },
                        )
                    )

    # 2. Extract narrative text blocks with bounding boxes via PyMuPDF
    pdf_doc = fitz.open(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page_idx in range(len(pdf_doc)):
        page_num = page_idx + 1
        page = pdf_doc[page_idx]
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

        page_text_parts = []
        block_bboxes = []
        for b in blocks:
            if b[6] == 0 and b[4].strip():  # text block only
                page_text_parts.append(b[4].strip())
                block_bboxes.append((round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)))

        full_page_text = "\n\n".join(page_text_parts)
        if not full_page_text.strip():
            continue

        chunks = splitter.split_text(full_page_text)
        for chunk in chunks:
            raw_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": doc_name,
                        "page": page_num,
                        "chunk_type": "text",
                        "bbox": block_bboxes[0] if block_bboxes else None,
                    },
                )
            )

    pdf_doc.close()
    return raw_documents