from pathlib import Path
from typing import List

import fitz
from langchain_core.runnables import RunnableLambda

from core.database import CivicDatabase
from generation.generator import CivicGroundedGenerator
from generation.schemas import Citation, GroundedCivicResponse
from retrieval.hybrid_retriever import HybridCivicRetriever
import retrieval.embeddings as emb_module
import retrieval.vector_store as vs_module
from services.rag_pipeline import CivicRAGService


class _OfflineEmbeddings:
    """Fast deterministic embeddings for end-to-end integration tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def __call__(self, text: str) -> List[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> List[float]:
        lower = text.lower()
        return [
            1.0 if "rythu" in lower or "farmer" in lower else 0.0,
            1.0 if "10,000" in lower or "acre" in lower else 0.0,
            0.5,
        ]


def _create_sample_scheme_pdf(pdf_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        (
            "Rythu Bandhu Agriculture Investment Support Scheme\n"
            "Eligible land-owning farmers receive Rs. 10,000 per acre per year "
            "across Kharif and Rabi seasons via Direct Benefit Transfer."
        ),
    )
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_end_to_end_ingest_deduplicate_and_query(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(emb_module, "get_embedding_model", lambda: _OfflineEmbeddings())
    monkeypatch.setattr(vs_module, "get_embedding_model", lambda: _OfflineEmbeddings())

    db = CivicDatabase(db_path=tmp_path / "civic.db")
    retriever = HybridCivicRetriever(index_dir=tmp_path / "indexes")
    generator = CivicGroundedGenerator(api_key="test-key")

    mocked_reply = GroundedCivicResponse(
        answer="Eligible farmers receive Rs. 10,000 per acre annually via DBT.",
        target_language="English",
        is_grounded=True,
        confidence_level="High",
        citations=[
            Citation(
                source="rythu_bandhu.pdf",
                page=1,
                quote="Eligible land-owning farmers receive Rs. 10,000 per acre per year",
            )
        ],
        missing_information="",
    )
    monkeypatch.setattr(
        generator,
        "_build_structured_llm",
        lambda: RunnableLambda(lambda _x: mocked_reply),
    )

    service = CivicRAGService(db=db, retriever=retriever, generator=generator)
    pdf_file = _create_sample_scheme_pdf(tmp_path / "rythu_bandhu.pdf")

    first = service.ingest_pdf(pdf_file)
    assert first.status == "indexed"
    assert first.page_count == 1
    assert first.chunk_count >= 1

    second = service.ingest_pdf(pdf_file)
    assert second.status == "duplicate"
    assert second.file_hash == first.file_hash

    outcome = service.ask(
        question="How much financial support per acre is given under Rythu Bandhu?",
        target_language="English",
    )
    assert outcome.response.is_grounded is True
    assert outcome.response.confidence_level == "High"
    assert len(outcome.retrieved_chunks) >= 1

    recent_logs = service.get_recent_query_logs()
    assert len(recent_logs) == 1
    assert recent_logs[0]["grounded"] is True
    assert recent_logs[0]["citations"][0]["source"] == "rythu_bandhu.pdf"

    stats = service.get_stats()
    assert stats["doc_count"] == 1
    assert stats["total_queries"] == 1