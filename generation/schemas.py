from typing import List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Exact document-and-page provenance for a factual claim."""

    source: str = Field(description="Exact PDF filename from the retrieved context.")
    page: int = Field(description="1-indexed page number supporting the claim.")
    quote: str = Field(
        description="Short verbatim snippet from the source supporting the statement."
    )


class GroundedCivicResponse(BaseModel):
    """Structured, citation-verified answer returned by Gemini."""

    answer: str = Field(
        description=(
            "Clear, citizen-friendly response in the requested target language. "
            "Must rely strictly on the provided context."
        )
    )
    target_language: str = Field(
        description="Language in which the answer is written (e.g., English, Hindi, Telugu)."
    )
    is_grounded: bool = Field(
        description=(
            "True if the retrieved documents contain enough evidence to answer the query; "
            "False if the documents lack the requested facts."
        )
    )
    confidence_level: str = Field(
        description="One of: High, Medium, Low, or Insufficient Evidence."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of exact source citations backing every factual claim.",
    )
    missing_information: str = Field(
        default="",
        description="Explicit note on what details are absent from the uploaded documents, if any.",
    )