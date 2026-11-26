import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from core.exceptions import GenerationError
from generation.generator import CivicGroundedGenerator
from generation.prompts import format_retrieved_chunks
from generation.schemas import Citation, GroundedCivicResponse
from retrieval.hybrid_retriever import RetrievedChunk


def test_format_retrieved_chunks_includes_provenance() -> None:
    chunks = [
        RetrievedChunk(
            document=Document(
                page_content="Annual assistance of Rs. 6,000 in three installments.",
                metadata={
                    "source": "pm_kisan.pdf",
                    "page": 2,
                    "content_type": "prose",
                },
            ),
            rrf_score=0.016393,
            dense_rank=1,
            lexical_rank=1,
        )
    ]
    formatted = format_retrieved_chunks(chunks)
    assert "Source: pm_kisan.pdf" in formatted
    assert "Page: 2" in formatted
    assert "Rs. 6,000" in formatted


def test_generator_empty_chunks_returns_ungrounded_fallback() -> None:
    generator = CivicGroundedGenerator(api_key="dummy-key")
    response = generator.generate(
        question="What is the maximum subsidy?",
        chunks=[],
        target_language="Hindi",
    )
    assert response.is_grounded is False
    assert response.confidence_level == "Insufficient Evidence"
    assert response.citations == []
    assert response.target_language == "Hindi"


def test_generator_raises_on_blank_question() -> None:
    generator = CivicGroundedGenerator(api_key="dummy-key")
    with pytest.raises(GenerationError, match="cannot be empty"):
        generator.generate(question="   ", chunks=[])


def test_generator_invokes_structured_chain_with_mocked_llm(monkeypatch) -> None:
    expected = GroundedCivicResponse(
        answer="Eligible farmers receive Rs. 6,000 per year [pm_kisan.pdf, p. 2].",
        target_language="English",
        is_grounded=True,
        confidence_level="High",
        citations=[
            Citation(
                source="pm_kisan.pdf",
                page=2,
                quote="Annual assistance of Rs. 6,000 in three installments.",
            )
        ],
        missing_information="",
    )

    generator = CivicGroundedGenerator(api_key="dummy-key")
    monkeypatch.setattr(
        generator,
        "_build_structured_llm",
        lambda: RunnableLambda(lambda _prompt_val: expected),
    )

    sample_chunk = RetrievedChunk(
        document=Document(
            page_content="Annual assistance of Rs. 6,000 in three installments.",
            metadata={"source": "pm_kisan.pdf", "page": 2, "content_type": "prose"},
        ),
        rrf_score=0.016,
    )
    result = generator.generate(
        question="How much money do farmers get?",
        chunks=[sample_chunk],
        target_language="English",
    )
    assert result.is_grounded is True
    assert result.confidence_level == "High"
    assert len(result.citations) == 1
    assert result.citations[0].page == 2