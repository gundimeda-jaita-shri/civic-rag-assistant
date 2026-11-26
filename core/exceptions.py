class CivicRAGError(Exception):
    """Base exception for the Civic Scheme RAG pipeline."""


class EmptyDocumentError(CivicRAGError):
    """Raised when an uploaded PDF contains no extractable text or tables."""


class MissingAPIKeyError(CivicRAGError):
    """Raised when GOOGLE_API_KEY is missing from environment variables."""


class RetrievalError(CivicRAGError):
    """Raised when vector or lexical index construction fails."""


class StorageError(CivicRAGError):
    """Raised when SQLite or FAISS persistence operations fail."""