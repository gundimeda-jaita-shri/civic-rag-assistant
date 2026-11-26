from datetime import datetime, timezone
from typing import List

from generation.schemas import GroundedCivicResponse
from retrieval.hybrid_retriever import RetrievedChunk


def build_markdown_audit_report(
    question: str,
    response: GroundedCivicResponse,
    retrieved_chunks: List[RetrievedChunk],
    latency_ms: int,
) -> str:
    """Generate a downloadable Markdown verification report for a civic Q&A response."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Civic Scheme & Document Verification Report",
        "",
        f"- **Generated At:** {timestamp}",
        f"- **Target Language:** {response.target_language}",
        f"- **Grounded Status:** {'Verified' if response.is_grounded else 'Insufficient Evidence'}",
        f"- **Confidence Level:** {response.confidence_level}",
        f"- **Pipeline Latency:** {latency_ms} ms",
        "",
        "## Citizen Query",
        f"> {question}",
        "",
        "## Verified Answer",
        response.answer,
        "",
    ]

    if response.missing_information:
        lines.extend(
            [
                "## Missing Information / Caveats",
                response.missing_information,
                "",
            ]
        )

    lines.append("## Supporting Citations")
    if response.citations:
        for idx, cite in enumerate(response.citations, start=1):
            lines.append(
                f"{idx}. **{cite.source}** (Page {cite.page}): *\"{cite.quote}\"*"
            )
    else:
        lines.append("_No direct citations attached._")

    lines.extend(["", "## Retrieved Evidence Chunks (Hybrid FAISS + BM25)"])
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        lines.extend(
            [
                f"### Evidence #{idx} — `{chunk.source}` (Page {chunk.page}, Score: `{chunk.score:.4f}`)",
                "```text",
                chunk.content.strip(),
                "```",
                "",
            ]
        )

    return "\n".join(lines)