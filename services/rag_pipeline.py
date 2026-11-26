from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from core.config import get_settings
from core.logger import get_logger
from database.sqlite_store import CivicDatabase
from generation.generator import CivicGroundedGenerator
from generation.schemas import GroundedCivicResponse
from ingestion.chunker import chunk_parsed_document
from ingestion.hashing import compute_sha256
from ingestion.pdf_extractor import parse_civic_pdf
from retrieval.hybrid_retriever import HybridCivicRetriever, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class IngestionOutcome:
    """Result summary returned after ingesting a civic PDF."""

    filename: str
    file_hash: str
    status: str  # "indexed" | "duplicate"
    page_count: int
    table_count: int
    chunk_count: int


@dataclass
class QueryOutcome:
    """Full pipeline output combining structured response and retrieved evidence."""

    question: str
    response: GroundedCivicResponse
    retrieved_chunks: List[RetrievedChunk]


class CivicRAGService:
    """End-to-end orchestrator for civic document ingestion, retrieval, and grounded QA."""

    def __init__(
        self,
        db: Optional[CivicDatabase] = None,
        retriever: Optional[HybridCivicRetriever] = None,
        generator: Optional[CivicGroundedGenerator] = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.db = db or CivicDatabase(db_path=settings.sqlite_db_path)
        self.retriever = retriever or HybridCivicRetriever(
            index_dir=settings.faiss_index_dir
        )
        self.generator = generator or CivicGroundedGenerator(
            model_name=settings.gemini_model,
            api_key=settings.google_api_key,
        )

    def ingest_pdf(self, file_path: Path) -> IngestionOutcome:
        """Validate, deduplicate via SHA-256, parse, chunk, and index a civic PDF."""
        path = Path(file_path)
        file_hash = compute_sha256(path)

        existing = self.db.get_document_by_hash(file_hash)
        if existing is not None:
            logger.info(
                "Skipping duplicate document '%s' (hash=%s).",
                path.name,
                file_hash[:12],
            )
            return IngestionOutcome(
                filename=existing["filename"],
                file_hash=file_hash,
                status="duplicate",
                page_count=existing["page_count"],
                table_count=existing["table_count"],
                chunk_count=existing["chunk_count"],
            )

        parsed_doc = parse_civic_pdf(path, max_size_mb=self.settings.max_upload_mb)
        chunks = chunk_parsed_document(
            parsed_doc,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        indexed_count = self.retriever.index_documents(chunks)

        self.db.record_document(
            filename=parsed_doc.filename,
            file_hash=parsed_doc.file_hash,
            page_count=parsed_doc.page_count,
            table_count=parsed_doc.table_count,
            chunk_count=indexed_count,
        )

        return IngestionOutcome(
            filename=parsed_doc.filename,
            file_hash=parsed_doc.file_hash,
            status="indexed",
            page_count=parsed_doc.page_count,
            table_count=parsed_doc.table_count,
            chunk_count=indexed_count,
        )

    def ask(
        self,
        question: str,
        target_language: str = "English",
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> QueryOutcome:
        """Retrieve relevant civic chunks, generate a grounded answer, and log the query."""
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k or self.settings.top_k_retrieval,
            source_filter=source_filter,
        )
        response = self.generator.generate(
            question=question,
            chunks=chunks,
            target_language=target_language,
        )
        self.db.log_query(
            question=question,
            target_language=target_language,
            is_grounded=response.is_grounded,
            confidence_level=response.confidence_level,
            citation_count=len(response.citations),
        )
        return QueryOutcome(
            question=question,
            response=response,
            retrieved_chunks=chunks,
        )

    def list_indexed_documents(self) -> List[Dict]:
        """Return metadata for all indexed civic PDFs."""
        return self.db.list_documents()

    def get_recent_query_logs(self, limit: int = 20) -> List[Dict]:
        """Return recent audit log entries."""
        return self.db.list_recent_queries(limit=limit)