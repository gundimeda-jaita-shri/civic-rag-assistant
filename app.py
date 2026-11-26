import html
import streamlit as st

from core.config import get_settings
from core.exceptions import CivicRAGError
from services.document_manager import CivicDocumentManager
from services.rag_pipeline import CivicRAGService
from ui.exporter import build_markdown_audit_report
from ui.styles import (
    SUPPORTED_LANGUAGES,
    format_confidence_badge,
    inject_civic_styles,
)

st.set_page_config(
    page_title="Civic Scheme & Document Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(inject_civic_styles(), unsafe_allow_html=True)


@st.cache_resource
def get_pipeline_components() -> tuple[CivicRAGService, CivicDocumentManager]:
    service = CivicRAGService()
    manager = CivicDocumentManager(db=service.db, retriever=service.retriever)
    return service, manager


service, doc_manager = get_pipeline_components()
settings = get_settings()

# Header Banner
st.markdown(
    """
    <div class="civic-header">
        <h1>🏛️ Multilingual Civic Scheme & Document Assistant</h1>
        <p>Citation-grounded public policy & welfare scheme assistant powered by Hybrid Retrieval (FAISS + BM25) and Gemini.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Document Ingestion, Scope Filter & Telemetry
with st.sidebar:
    st.subheader("📄 Civic Document Registry")
    uploaded_files = st.file_uploader(
        "Upload Official Scheme PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum {settings.max_upload_mb} MB per PDF. Duplicates are automatically skipped via SHA-256.",
    )

    if uploaded_files and st.button("Index Uploaded PDFs", type="primary", use_container_width=True):
        for uploaded in uploaded_files:
            try:
                saved_path = doc_manager.save_uploaded_bytes(
                    uploaded.name, uploaded.getvalue()
                )
                outcome = service.ingest_pdf(saved_path)
                if outcome.status == "indexed":
                    st.success(
                        f"Indexed **{outcome.filename}** ({outcome.page_count} pages, "
                        f"{outcome.table_count} tables, {outcome.chunk_count} chunks)"
                    )
                else:
                    st.info(f"Skipped duplicate: **{outcome.filename}**")
            except CivicRAGError as exc:
                st.error(f"Failed to index {uploaded.name}: {exc}")

    st.divider()
    st.subheader("⚙️ Query Controls")
    target_language = st.selectbox(
        "Response Language",
        options=SUPPORTED_LANGUAGES,
        index=0,
    )

    indexed_docs = service.list_indexed_documents()
    doc_names = [d["filename"] for d in indexed_docs]
    selected_scope = st.selectbox(
        "Document Scope Filter",
        options=["All Indexed Documents"] + doc_names,
        index=0,
    )
    source_filter = None if selected_scope == "All Indexed Documents" else selected_scope

    st.divider()
    st.subheader("📚 Indexed Knowledge Base")
    if not indexed_docs:
        st.caption("No PDFs indexed yet. Upload a scheme circular above to begin.")
    else:
        for doc in indexed_docs:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(
                    f"**{doc['filename']}**  \n"
                    f"<small>{doc['page_count']}p · {doc['chunk_count']} chunks · {doc['table_count']} tables</small>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Remove {doc['filename']}"):
                    try:
                        doc_manager.remove_document_and_rebuild(doc["filename"])
                        service.retriever = doc_manager.retriever
                        st.rerun()
                    except CivicRAGError as exc:
                        st.error(str(exc))

    st.divider()
    stats = service.get_stats()
    m1, m2 = st.columns(2)
    m1.metric("Documents", stats["doc_count"])
    m2.metric("Total Chunks", stats["total_chunks"])
    m3, m4 = st.columns(2)
    m3.metric("Queries", stats["total_queries"])
    m4.metric("Avg Latency", f"{stats['avg_latency_ms']} ms")

# Main Tabs: Grounded Q&A vs. Audit Log
tab_qa, tab_audit = st.tabs(["🔍 Ask & Verify", "📜 Query Audit Log"])

with tab_qa:
    if not settings.google_api_key:
        st.warning(
            "`GOOGLE_API_KEY` is not set in your `.env` file. Add your Gemini API key before running queries."
        )

    question = st.text_area(
        "Enter your question about eligibility, benefits, required documents, or deadlines:",
        placeholder="e.g., What is the annual financial assistance per acre under Rythu Bandhu, and who is eligible?",
        height=100,
    )

    if st.button("Generate Verified Answer", type="primary", disabled=not question.strip()):
        if not indexed_docs:
            st.warning("Please upload and index at least one civic PDF in the sidebar first.")
        else:
            with st.spinner("Retrieving evidence (FAISS + BM25) and verifying citations..."):
                try:
                    outcome = service.ask(
                        question=question.strip(),
                        target_language=target_language,
                        source_filter=source_filter,
                    )
                    st.session_state["last_outcome"] = outcome
                except CivicRAGError as exc:
                    st.error(f"Pipeline Error: {exc}")

    if "last_outcome" in st.session_state:
        outcome = st.session_state["last_outcome"]
        resp = outcome.response

        badge = format_confidence_badge(resp.confidence_level)
        c1, c2, c3 = st.columns([2, 2, 1])
        c1.markdown(f"**Verification Status:** {badge}")
        c2.markdown(f"**Language:** `{resp.target_language}`")
        c3.markdown(f"**Latency:** `{outcome.latency_ms} ms`")

        st.markdown("### Verified Response")
        st.markdown(resp.answer)

        if resp.missing_information:
            st.info(f"**Missing Information / Caveats:** {resp.missing_information}")

        st.markdown("### Supporting Citations")
        if resp.citations:
            for idx, cite in enumerate(resp.citations, start=1):
                safe_source = html.escape(cite.source)
                safe_quote = html.escape(cite.quote)
                st.markdown(
                    f"""
                    <div class="citation-box">
                        <div class="citation-meta">Citation #{idx} — {safe_source} (Page {cite.page})</div>
                        <div class="citation-quote">"{safe_quote}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No direct verbatim citations were attached to this response.")

        report_md = build_markdown_audit_report(
            question=outcome.question,
            response=resp,
            retrieved_chunks=outcome.retrieved_chunks,
            latency_ms=outcome.latency_ms,
        )
        st.download_button(
            label="📥 Download Verification Audit Report (.md)",
            data=report_md,
            file_name="civic_verification_report.md",
            mime="text/markdown",
        )

        with st.expander("🔎 Inspect Raw Hybrid Retrieval Evidence (FAISS + BM25)"):
            for idx, chunk in enumerate(outcome.retrieved_chunks, start=1):
                st.markdown(
                    f"**Evidence #{idx}** — `{chunk.source}` · Page `{chunk.page}` · "
                    f"RRF Score: `{chunk.score:.4f}`"
                )
                st.code(chunk.content, language="text")

with tab_audit:
    st.subheader("Recent Citizen Query Audit Trail")
    logs = service.get_recent_query_logs(limit=25)
    if not logs:
        st.caption("No queries logged yet.")
    else:
        for entry in logs:
            status_icon = "✅" if entry["grounded"] else "⚠️"
            with st.expander(
                f"{status_icon} [{entry['language']}] {entry['question'][:90]} ({entry['latency_ms']} ms)"
            ):
                st.markdown(f"**Timestamp:** `{entry['created_at']}`")
                st.markdown(f"**Answer:** {entry['answer']}")
                if entry["citations"]:
                    st.markdown("**Citations:**")
                    for c in entry["citations"]:
                        st.markdown(f"- `{c['source']}` (p.{c['page']}): *\"{c['quote']}\"*")