from dataclasses import dataclass
from typing import List, Sequence

from generation.schemas import GroundedCivicResponse
from retrieval.hybrid_retriever import RetrievedChunk


@dataclass
class QueryEvalResult:
    """Evaluation outcome for a single benchmark query."""

    query_id: str
    language: str
    recall_at_k: float
    reciprocal_rank: float
    grounding_match: bool
    keyword_coverage: float


def compute_recall_and_mrr(
    retrieved_chunks: Sequence[RetrievedChunk],
    expected_source: str,
    expected_page: int,
) -> tuple[float, float]:
    """Compute binary Recall@K and Reciprocal Rank (MRR) for expected source + page."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.document.metadata or {}
        if meta.get("source") == expected_source and int(meta.get("page", -1)) == expected_page:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def compute_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """Fraction of expected domain keywords present (case-insensitive) in the answer."""
    if not expected_keywords:
        return 1.0
    lower_answer = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_answer)
    return hits / len(expected_keywords)


def evaluate_single_prediction(
    query_id: str,
    language: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    response: GroundedCivicResponse,
    expected_source: str,
    expected_page: int,
    expected_keywords: List[str],
    should_be_grounded: bool,
) -> QueryEvalResult:
    """Score retrieval and generation outputs for one benchmark sample."""
    if should_be_grounded:
        recall, mrr = compute_recall_and_mrr(
            retrieved_chunks, expected_source, expected_page
        )
        kw_cov = compute_keyword_coverage(response.answer, expected_keywords)
    else:
        recall, mrr, kw_cov = 1.0, 1.0, 1.0

    grounding_match = response.is_grounded == should_be_grounded
    return QueryEvalResult(
        query_id=query_id,
        language=language,
        recall_at_k=recall,
        reciprocal_rank=mrr,
        grounding_match=grounding_match,
        keyword_coverage=kw_cov,
    )