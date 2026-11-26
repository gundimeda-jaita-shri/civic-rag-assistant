from typing import List
from langchain_core.prompts import ChatPromptTemplate

from retrieval.hybrid_retriever import RetrievedChunk

CIVIC_GROUNDED_SYSTEM_PROMPT = """You are a verifiable, citizen-centric Civic Scheme & Public Document Assistant.

STRICT GROUNDING & ANTI-HALLUCINATION RULES:
1. Answer ONLY using facts explicitly stated in the <retrieved_context> blocks below.
2. NEVER invent eligibility limits, subsidy amounts, deadlines, application URLs, or legal rules not present in the context.
3. If the <retrieved_context> does not contain enough information to answer the user's question:
   - Set `is_grounded` to `false`.
   - Set `confidence_level` to `"Insufficient Evidence"`.
   - Clearly explain in the requested target language what information is missing from the uploaded documents.
4. Every factual statement when `is_grounded` is `true` MUST include at least one exact citation (`source`, `page`, and a short verbatim `quote` from the chunk).
5. Preserve exact numbers, currency figures (e.g., Rs., ₹), income thresholds, and Markdown tables accurately.
6. Write the final `answer` and `missing_information` naturally in the requested `target_language`, even if the source documents are in English or another Indian language.
"""

CIVIC_GROUNDED_USER_PROMPT = """Target Response Language: {target_language}

User Question:
{question}

<retrieved_context>
{formatted_context}
</retrieved_context>
"""


def build_civic_prompt_template() -> ChatPromptTemplate:
    """Return the system + user ChatPromptTemplate for grounded civic synthesis."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", CIVIC_GROUNDED_SYSTEM_PROMPT),
            ("human", CIVIC_GROUNDED_USER_PROMPT),
        ]
    )


def format_retrieved_chunks(chunks: List[RetrievedChunk]) -> str:
    """Format RRF-ranked chunks with explicit source, page, and type headers for LLM grounding."""
    if not chunks:
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    blocks: List[str] = []
    for idx, item in enumerate(chunks, start=1):
        meta = item.document.metadata
        source = meta.get("source", "unknown.pdf")
        page = meta.get("page", "?")
        ctype = meta.get("content_type", "prose")
        blocks.append(
            f"[Chunk {idx} | Source: {source} | Page: {page} | Type: {ctype} | RRF: {item.rrf_score}]\n"
            f"{item.document.page_content}"
        )
    return "\n\n---\n\n".join(blocks)