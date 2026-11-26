from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from core.config import get_settings
from core.logger import get_logger
from retrieval.bm25_store import CivicBM25Store
from retrieval.vector_store import CivicVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a fused search hit with provenance and RRF confidence score."""

    document: Document
    rrf_score: float
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None


class HybridCivicRetriever:
    """Fuses dense FAISS semantic hits and sparse BM25 lexical hits via Reciprocal Rank Fusion."""

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        lexical_weight: float = 0.4,
    ) -> None:
        self.vector_store = CivicVectorStore(index_dir=index_dir)
        self.bm25_store = CivicBM25Store(index_dir=index_dir)
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def index_documents(self, documents: List[Document]) -> int:
        """Index chunks into both dense FAISS and sparse BM25 stores."""
        if not documents:
            return 0
        self.vector_store.add_documents(documents)
        self.bm25_store.add_documents(documents)
        return len(documents)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Execute hybrid retrieval and return top-k RRF-ranked civic document chunks."""
        settings = get_settings()
        final_k = top_k or settings.top_k_retrieval
        candidate_k = max(final_k * 3, 15)

        dense_hits = self.vector_store.similarity_search_with_scores(query, k=candidate_k)
        lexical_hits = self.bm25_store.search_with_scores(query, k=candidate_k)

        fused_scores: Dict[str, float] = {}
        doc_registry: Dict[str, Document] = {}
        dense_ranks: Dict[str, int] = {}
        lexical_ranks: Dict[str, int] = {}

        for rank, (doc, _dist) in enumerate(dense_hits, start=1):
            if source_filter and doc.metadata.get("source") != source_filter:
                continue
            cid = doc.metadata.get("chunk_id") or f"dense_{rank}_{hash(doc.page_content)}"
            doc_registry[cid] = doc
            dense_ranks[cid] = rank
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )

        for rank, (doc, _bm25_score) in enumerate(lexical_hits, start=1):
            if source_filter and doc.metadata.get("source") != source_filter:
                continue
            cid = doc.metadata.get("chunk_id") or f"lex_{rank}_{hash(doc.page_content)}"
            doc_registry[cid] = doc
            lexical_ranks[cid] = rank
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (
                self.lexical_weight / (self.rrf_k + rank)
            )

        ranked_ids = sorted(
            fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True
        )

        results: List[RetrievedChunk] = [
            RetrievedChunk(
                document=doc_registry[cid],
                rrf_score=round(fused_scores[cid], 6),
                dense_rank=dense_ranks.get(cid),
                lexical_rank=lexical_ranks.get(cid),
            )
            for cid in ranked_ids[:final_k]
        ]
        logger.info(
            "Hybrid retrieval for query '%s' returned %d chunks (dense=%d, lexical=%d).",
            query[:60],
            len(results),
            len(dense_hits),
            len(lexical_hits),
        )
        return results

    def reset_all(self) -> None:
        """Clear both FAISS and BM25 indexes."""
        self.vector_store.reset()
        self.bm25_store.reset()