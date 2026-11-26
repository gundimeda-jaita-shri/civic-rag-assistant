from pathlib import Path
from typing import List

import fitz

from core.database import CivicDatabase
from retrieval.hybrid_retriever import HybridCivicRetriever
import retrieval.embeddings as emb_module
import retrieval.vector_store as vs_module
from services.document_manager import CivicDocumentManager
from services.rag_pipeline import CivicRAGService
from generation.generator import CivicGroundedGenerator


class _StubEmbeddings:
    """Fast offline embeddings for document lifecycle tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t) % 7), 1.0, 0.25] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text) % 7), 1.0, 0.25]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


def _write_pdf(path: Path, body: str) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), body)
    doc.save(path)
    doc.close()
    return path


def test_remove_document_rebuilds_indexes_cleanly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(emb_module, "get_embedding_model", lambda: _StubEmbeddings())
    monkeypatch.setattr(vs_module, "get_embedding_model", lambda: _StubEmbeddings())

    db = CivicDatabase(db_path=tmp_path / "civic.db")
    retriever = HybridCivicRetriever(index_dir=tmp_path / "indexes")
    upload_dir = tmp_path / "uploads"
    manager = CivicDocumentManager(
        db=db, retriever=retriever, upload_dir=upload_dir
    )
    service = CivicRAGService(
        db=db,
        retriever=retriever,
        generator=CivicGroundedGenerator(api_key="dummy-key"),
    )

    pdf_a = _write_pdf(upload_dir / "scheme_a.pdf", "Scheme A provides housing subsidy.")
    pdf_b = _write_pdf(upload_dir / "scheme_b.pdf", "Scheme B provides crop insurance.")

    service.ingest_pdf(pdf_a)
    service.ingest_pdf(pdf_b)
    assert len(db.list_documents()) == 2

    remaining_chunks = manager.remove_document_and_rebuild("scheme_a.pdf")
    assert remaining_chunks >= 1
    assert not pdf_a.exists()
    assert pdf_b.exists()

    docs = db.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "scheme_b.pdf"