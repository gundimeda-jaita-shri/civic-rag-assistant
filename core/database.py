import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.config import settings
from core.exceptions import StorageError


class CivicDatabase:
    """SQLite persistence layer for indexed civic documents and Q&A audit logs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        page_count INTEGER NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        table_count INTEGER NOT NULL DEFAULT 0,
                        uploaded_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS query_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT NOT NULL,
                        language TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        citations_json TEXT NOT NULL,
                        grounded INTEGER NOT NULL,
                        latency_ms INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to initialize SQLite schema: {exc}") from exc

    def upsert_document(
        self,
        filename: str,
        file_hash: str,
        page_count: int,
        chunk_count: int,
        table_count: int = 0,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (filename, file_hash, page_count, chunk_count, table_count, uploaded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(filename) DO UPDATE SET
                        file_hash=excluded.file_hash,
                        page_count=excluded.page_count,
                        chunk_count=excluded.chunk_count,
                        table_count=excluded.table_count,
                        uploaded_at=excluded.uploaded_at
                    """,
                    (filename, file_hash, page_count, chunk_count, table_count, now),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to save document metadata: {exc}") from exc

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    def log_query(
        self,
        question: str,
        language: str,
        answer: str,
        citations: List[Dict[str, Any]],
        grounded: bool,
        latency_ms: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO query_logs (question, language, answer, citations_json, grounded, latency_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        question,
                        language,
                        answer,
                        json.dumps(citations, ensure_ascii=False),
                        1 if grounded else 0,
                        latency_ms,
                        now,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to record query audit log: {exc}") from exc

    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["citations"] = json.loads(item.pop("citations_json"))
                item["grounded"] = bool(item["grounded"])
                results.append(item)
            return results
    def document_exists_by_hash(self, file_hash: str) -> Optional[str]:
        """Return the filename if an identical SHA-256 PDF is already registered."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT filename FROM documents WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return str(row["filename"]) if row else None

    def delete_document(self, filename: str) -> None:
        """Remove a document record from the SQLite registry."""
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to delete document record: {exc}") from exc

    def get_registry_stats(self) -> Dict[str, Any]:
        """Compute aggregate counts and latency telemetry for the dashboard."""
        with self._connect() as conn:
            doc_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS doc_count,
                    COALESCE(SUM(page_count), 0) AS total_pages,
                    COALESCE(SUM(chunk_count), 0) AS total_chunks,
                    COALESCE(SUM(table_count), 0) AS total_tables
                FROM documents
                """
            ).fetchone()

            query_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_queries,
                    COALESCE(SUM(grounded), 0) AS grounded_queries,
                    COALESCE(ROUND(AVG(latency_ms)), 0) AS avg_latency_ms
                FROM query_logs
                """
            ).fetchone()

            return {
                "doc_count": int(doc_row["doc_count"]),
                "total_pages": int(doc_row["total_pages"]),
                "total_chunks": int(doc_row["total_chunks"]),
                "total_tables": int(doc_row["total_tables"]),
                "total_queries": int(query_row["total_queries"]),
                "grounded_queries": int(query_row["grounded_queries"]),
                "avg_latency_ms": int(query_row["avg_latency_ms"]),
            }