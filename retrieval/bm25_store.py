import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.config import get_settings
from core.exceptions import RetrievalError
from core.logger import get_logger

logger = get_logger(__name__)


def _tokenize_multilingual(text: str) -> List[str]:
    """Tokenize Unicode words and numbers across English and Indic scripts."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class CivicBM25Store:
    """Manages a persisted lexical BM25 index for exact scheme codes, numbers, and keywords."""

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        base_dir = Path(index_dir or settings.faiss_index_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = base_dir / "bm25_corpus.pkl"
        self._documents: List[Document] = []
        self._bm25: Optional[BM25Okapi] = None

    def _rebuild_index(self) -> None:
        if not self._documents:
            self._bm25 = None
            return
        tokenized = [_tokenize_multilingual(doc.page_content) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    def load(self) -> bool:
        """Load persisted documents from disk and build the in-memory BM25 index."""
        if self._bm25 is not None:
            return True

        if not self.corpus_path.exists():
            return False

        try:
            with self.corpus_path.open("rb") as f:
                self._documents = pickle.load(f)
            self._rebuild_index()
            logger.info(
                "Loaded %d documents into BM25 lexical index from '%s'.",
                len(self._documents),
                self.corpus_path,
            )
            return self._bm25 is not None
        except Exception as exc:
            raise RetrievalError(f"Failed to load BM25 corpus: {exc}") from exc

    def add_documents(self, documents: List[Document]) -> int:
        """Append new chunks to the persisted corpus and rebuild the BM25 index."""
        if not documents:
            return 0

        try:
            if not self._documents and self.corpus_path.exists():
                self.load()

            self._documents.extend(documents)
            self._rebuild_index()

            with self.corpus_path.open("wb") as f:
                pickle.dump(self._documents, f)

            logger.info(
                "Persisted %d total chunks to BM25 corpus at '%s'.",
                len(self._documents),
                self.corpus_path,
            )
            return len(documents)
        except Exception as exc:
            raise RetrievalError(f"Failed to update BM25 index: {exc}") from exc

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Return top-k lexical matches with positive BM25 scores."""
        if not self.load() or self._bm25 is None:
            return []

        query_tokens = _tokenize_multilingual(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda idx: scores[idx], reverse=True
        )

        results: List[Tuple[Document, float]] = []
        for idx in ranked_indices[:k]:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self._documents[idx], score))
        return results

    def reset(self) -> None:
        """Clear in-memory documents and delete the persisted BM25 corpus."""
        self._documents = []
        self._bm25 = None
        if self.corpus_path.exists():
            self.corpus_path.unlink()
        logger.info("Cleared BM25 corpus at '%s'.", self.corpus_path)