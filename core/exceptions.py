class CivicRAGError(Exception):
    """Base exception for the Multilingual Civic Scheme & Document Assistant."""


class ConfigurationError(CivicRAGError):
    """Raised when required environment variables or settings are invalid."""


class StorageError(CivicRAGError):
    """Raised when SQLite persistence or audit logging fails."""


class DatabaseError(StorageError):
    """Alias for SQLite persistence or audit logging errors."""


class IngestionError(CivicRAGError):
    """Raised when PDF validation, parsing, or chunking fails."""


class DocumentIngestionError(IngestionError):
    """Alias for PDF validation, parsing, or chunking errors."""


class RetrievalError(CivicRAGError):
    """Raised when FAISS/BM25 indexing or hybrid search fails."""


class GenerationError(CivicRAGError):
    """Raised when grounded LLM synthesis or verification fails."""