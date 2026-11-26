from langchain_core.documents import Document

from eval.metrics import (
    compute_keyword_coverage,
    compute_recall_and_mrr,
    evaluate_single_prediction,
)
from generation.schemas import Citation, GroundedCivicResponse
from retrieval.hybrid_retriever import RetrievedChunk


def test_compute_recall_and_mrr_ranks_correctly() -> None:
    chunks = [
        RetrievedChunk(
            document=Document(
                page_content="Other doc",
                metadata={"source": "other.pdf", "page": 2},
            ),
            rrf_score=0.016,
        ),
        RetrievedChunk(
            document=Document(
                page_content="Target doc",
                metadata={"source": "rythu_bandhu.pdf", "page": 1},
            ),
            rrf_score=0.015,
        ),
    ]
    recall, mrr = compute_recall_and_mrr(chunks, "rythu_bandhu.pdf", 1)
    assert recall == 1.0
    assert mrr == 0.5

    miss_recall, miss_mrr = compute_recall_and_mrr(chunks, "missing.pdf", 1)
    assert miss_recall == 0.0
    assert miss_mrr == 0.0


def test_evaluate_single_prediction_grounded_and_abstention() -> None:
    chunks = [
        RetrievedChunk(
            document=Document(
                page_content="Rs. 10,000 per acre for Kharif and Rabi.",
                metadata={"source": "rythu_bandhu.pdf", "page": 1},
            ),
            rrf_score=0.032,
        )
    ]
    grounded_resp = GroundedCivicResponse(
        answer="Farmers receive Rs. 10,000 per acre across Kharif and Rabi seasons.",
        target_language="English",
        is_grounded=True,
        confidence_level="High",
        citations=[
            Citation(
                source="rythu_bandhu.pdf",
                page=1,
                quote="Rs. 10,000 per acre for Kharif and Rabi.",
            )
        ],
        missing_information="",
    )

    scored = evaluate_single_prediction(
        query_id="q1",
        language="English",
        retrieved_chunks=chunks,
        response=grounded_resp,
        expected_source="rythu_bandhu.pdf",
        expected_page=1,
        expected_keywords=["10,000", "acre", "Kharif"],
        should_be_grounded=True,
    )
    assert scored.recall_at_k == 1.0
    assert scored.reciprocal_rank == 1.0
    assert scored.grounding_match is True
    assert scored.keyword_coverage == 1.0

    abstention_resp = GroundedCivicResponse(
        answer="Insufficient evidence in the uploaded documents.",
        target_language="English",
        is_grounded=False,
        confidence_level="Insufficient Evidence",
        citations=[],
        missing_information="No mention of Mars tractor subsidies.",
    )
    abstained = evaluate_single_prediction(
        query_id="q2",
        language="English",
        retrieved_chunks=chunks,
        response=abstention_resp,
        expected_source="rythu_bandhu.pdf",
        expected_page=1,
        expected_keywords=[],
        should_be_grounded=False,
    )
    assert abstained.grounding_match is True
    assert compute_keyword_coverage("No match", ["10,000"]) == 0.0