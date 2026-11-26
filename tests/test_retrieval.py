from pathlib import Path
from typing import List
from langchain_core.documents import Document

from retrieval.bm25_store import CivicBM25Store
from retrieval.hybrid_retriever import HybridCivicRetriever
import retrieval.embeddings as emb_module


class _DummyEmbeddings:
    """Deterministic lightweight embeddings for fast offline unit testing."""

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
            1.0 if "farmer" in lower or "kisan" in lower else 0.0,
            1.0 if "pension" in lower or "elderly" in lower else 0.0,
            1.0 if "scholarship" in lower or "student" in lower else 0.0,
            float(len(lower) % 10) / 10.0,
        ]


def test_bm25_multilingual_tokenization_and_search(tmp_path: Path) -> None:
    store = CivicBM25Store(index_dir=tmp_path)
    docs = [
        Document(
            page_content="PM-KISAN scheme provides Rs. 6000 per year to eligible farmers.",
            metadata={"source": "kisan.pdf", "page": 1, "chunk_id": "k1"},
        ),
        Document(
            page_content="प्रधानमंत्री आवास योजना ग्रामीण के तहत पक्का मकान सहायता।",
            metadata={"source": "awas.pdf", "page": 2, "chunk_id": "a1"},
        ),
    ]
    assert store.add_documents(docs) == 2

    hits = store.search_with_scores("6000 farmers", k=2)
    assert len(hits) >= 1
    assert hits[0][0].metadata["chunk_id"] == "k1"

    hindi_hits = store.search_with_scores("आवास योजना", k=2)
    assert len(hindi_hits) >= 1
    assert hindi_hits[0][0].metadata["chunk_id"] == "a1"


def test_hybrid_retriever_fuses_dense_and_lexical_results(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(emb_module, "get_embedding_model", lambda: _DummyEmbeddings())
    import retrieval.vector_store as vs_module

    monkeypatch.setattr(vs_module, "get_embedding_model", lambda: _DummyEmbeddings())

    retriever = HybridCivicRetriever(index_dir=tmp_path)
    docs = [
        Document(
            page_content="National Old Age Pension Scheme for senior citizens above 60.",
            metadata={"source": "pension.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content="Post-Matric Scholarship for SC/ST students with income below 2.5 lakh.",
            metadata={"source": "edu.pdf", "page": 3, "chunk_id": "e1"},
        ),
    ]
    assert retriever.index_documents(docs) == 2

    results = retriever.retrieve("elderly pension 60", top_k=2)
    assert len(results) >= 1
    assert results[0].document.metadata["chunk_id"] == "p1"
    assert results[0].rrf_score > 0