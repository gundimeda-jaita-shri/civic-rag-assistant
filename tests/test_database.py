from pathlib import Path
from core.database import CivicDatabase


def test_document_upsert_lookup_and_delete(tmp_path: Path) -> None:
    db = CivicDatabase(db_path=tmp_path / "test_civic.db")

    db.upsert_document(
        filename="pm_kisan_2026.pdf",
        file_hash="abc123sha256",
        page_count=12,
        chunk_count=34,
        table_count=3,
    )
    assert db.document_exists_by_hash("abc123sha256") == "pm_kisan_2026.pdf"
    assert db.document_exists_by_hash("unknown_hash") is None

    docs = db.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "pm_kisan_2026.pdf"
    assert docs[0]["table_count"] == 3

    db.delete_document("pm_kisan_2026.pdf")
    assert db.list_documents() == []


def test_query_audit_logging_and_stats(tmp_path: Path) -> None:
    db = CivicDatabase(db_path=tmp_path / "test_civic.db")

    db.upsert_document("scheme.pdf", "hash1", page_count=5, chunk_count=10, table_count=1)
    db.log_query(
        question="What is the annual benefit?",
        language="English",
        answer="Rs. 6,000 per year [1].",
        citations=[{"source": "scheme.pdf", "page": 2, "quote": "Rs. 6,000 per year"}],
        grounded=True,
        latency_ms=420,
    )

    history = db.get_recent_queries(limit=10)
    assert len(history) == 1
    assert history[0]["grounded"] is True
    assert history[0]["citations"][0]["page"] == 2

    stats = db.get_registry_stats()
    assert stats["doc_count"] == 1
    assert stats["total_queries"] == 1
    assert stats["grounded_queries"] == 1
    assert stats["avg_latency_ms"] == 420