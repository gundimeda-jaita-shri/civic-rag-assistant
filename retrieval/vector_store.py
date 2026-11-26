from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.config import get_settings
from core.exceptions import RetrievalError
from core.logger import get_logger
from retrieval.embeddings import get_embedding_model

logger = get_logger(__name__)


class CivicVectorStore:
    """Manages persistent FAISS storage and dense similarity search for civic chunks."""

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.faiss_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store: Optional[FAISS] = None

    @property
    def _index_file(self) -> Path:
        return self.index_dir / "index.faiss"

    def load(self) -> Optional[FAISS]:
        """Load an existing FAISS index from disk if available."""
        if self._store is not None:
            return self._store

        if not self._index_file.exists():
            return None

        try:
            embeddings = get_embedding_model()
            self._store = FAISS.load_local(
                str(self.index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index from '%s'.", self.index_dir)
            return self._store
        except Exception as exc:
            raise RetrievalError(f"Failed to load FAISS index: {exc}") from exc

    def add_documents(self, documents: List[Document]) -> int:
        """Embed and persist new LangChain Documents into the FAISS index."""
        if not documents:
            return 0

        try:
            embeddings = get_embedding_model()
            existing = self.load()
            if existing is None:
                self._store = FAISS.from_documents(documents, embeddings)
            else:
                existing.add_documents(documents)
                self._store = existing

            self._store.save_local(str(self.index_dir))
            logger.info(
                "Persisted %d chunks to FAISS index at '%s'.",
                len(documents),
                self.index_dir,
            )
            return len(documents)
        except Exception as exc:
            raise RetrievalError(f"Failed to index documents into FAISS: {exc}") from exc

    def similarity_search_with_scores(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Return top-k dense matches alongside L2 distance scores."""
        store = self.load()
        if store is None:
            return []

        try:
            return store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise RetrievalError(f"FAISS similarity search failed: {exc}") from exc

    def reset(self) -> None:
        """Clear in-memory state and delete persisted FAISS files."""
        self._store = None
        for file_path in self.index_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        logger.info("Cleared FAISS index directory '%s'.", self.index_dir)