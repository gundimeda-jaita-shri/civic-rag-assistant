import shutil
from pathlib import Path
from typing import Dict, List, Optional

from core.config import get_settings
from core.database import CivicDatabase
from core.logger import get_logger
from ingestion.chunker import chunk_parsed_document
from ingestion.pdf_extractor import parse_civic_pdf
from retrieval.hybrid_retriever import HybridCivicRetriever

logger = get_logger(__name__)


class CivicDocumentManager:
    """Manages uploaded civic PDF files, SQLite registry entries, and index rebuilds."""

    def __init__(
        self,
        db: Optional[CivicDatabase] = None,
        retriever: Optional[HybridCivicRetriever] = None,
        upload_dir: Optional[Path] = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.db = db or CivicDatabase(db_path=settings.sqlite_db_path)
        self.retriever = retriever or HybridCivicRetriever(
            index_dir=settings.faiss_index_dir
        )
        self.upload_dir = Path(upload_dir or settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _get_index_dir(self) -> Path:
        if hasattr(self.retriever, "index_dir"):
            return Path(self.retriever.index_dir)
        if hasattr(self.retriever, "vector_store") and hasattr(
            self.retriever.vector_store, "index_dir"
        ):
            return Path(self.retriever.vector_store.index_dir)
        return Path(self.settings.faiss_index_dir)

    def save_uploaded_bytes(self, filename: str, content: bytes) -> Path:
        """Persist raw uploaded PDF bytes into the managed upload directory."""
        safe_name = Path(filename).name
        target_path = self.upload_dir / safe_name
        target_path.write_bytes(content)
        logger.info("Saved uploaded PDF '%s' (%d bytes).", safe_name, len(content))
        return target_path

    def remove_document_and_rebuild(self, filename: str) -> int:
        """Delete a document from SQLite + disk and rebuild FAISS/BM25 from remaining PDFs."""
        safe_name = Path(filename).name
        self.db.delete_document(safe_name)

        stored_pdf = self.upload_dir / safe_name
        if stored_pdf.exists():
            stored_pdf.unlink()

        return self.rebuild_indexes_from_disk()

    def rebuild_indexes_from_disk(self) -> int:
        """Wipe vector/BM25 index files and re-index all remaining PDFs in SQLite."""
        index_dir = self._get_index_dir()
        if index_dir.exists():
            shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        self.retriever = HybridCivicRetriever(index_dir=index_dir)

        remaining_docs: List[Dict] = self.db.list_documents()
        total_chunks = 0
        for doc_row in remaining_docs:
            pdf_path = self.upload_dir / doc_row["filename"]
            if not pdf_path.exists():
                logger.warning(
                    "PDF '%s' registered in SQLite but missing on disk; skipping rebuild.",
                    doc_row["filename"],
                )
                continue
            parsed = parse_civic_pdf(pdf_path, max_size_mb=self.settings.max_upload_mb)
            chunks = chunk_parsed_document(
                parsed,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            indexed = self.retriever.index_documents(chunks)
            total_chunks += indexed

        logger.info(
            "Rebuilt hybrid indexes across %d documents (%d total chunks).",
            len(remaining_docs),
            total_chunks,
        )
        return total_chunks