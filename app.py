import os
import glob
import base64
from datetime import datetime
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(
    page_title="Multilingual Civic Scheme & Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Permanent Local Emblem Loader (Supports .in-, .in-*, .png, .jpg, .webp, .svg)
def detect_mime_type(raw_bytes: bytes, filename: str) -> str:
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    if b"<svg" in raw_bytes[:512].lower() or filename.lower().endswith(".svg"):
        return "image/svg+xml"
    return "image/png"


@st.cache_data
def get_permanent_emblem_src() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_patterns = [
        os.path.join(base_dir, "*.in_*"),
        os.path.join(base_dir, "*.in_"),
        os.path.join(base_dir, "india_emblem*"),
        os.path.join(base_dir, "emblem*"),
        os.path.join(base_dir, "assets", "*.in_*"),
        os.path.join(base_dir, "assets", "india_emblem*"),
    ]

    for pattern in search_patterns:
        matches = sorted(glob.glob(pattern))
        for file_path in matches:
            if os.path.isfile(file_path):
                with open(file_path, "rb") as img_file:
                    raw_bytes = img_file.read()
                if raw_bytes:
                    mime_type = detect_mime_type(raw_bytes, file_path)
                    b64_encoded = base64.b64encode(raw_bytes).decode("utf-8")
                    return f"data:{mime_type};base64,{b64_encoded}"

    # Formal inline SVG fallback so the header never breaks even if the file is moved
    fallback_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">
        <rect width="120" height="120" rx="8" fill="#FFFFFF"/>
        <circle cx="60" cy="60" r="46" fill="none" stroke="#1B365D" stroke-width="5"/>
        <circle cx="60" cy="60" r="10" fill="#1B365D"/>
        <text x="60" y="102" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#1B365D" text-anchor="middle">GOVT OF INDIA</text>
    </svg>
    """.strip()
    b64_svg = base64.b64encode(fallback_svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64_svg}"

# 3. Initialize Session State
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# 4. Load Multilingual Embedding Model (Cached)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embedder = load_embedder()

# 5. Sidebar Controls (Formal Display Settings, Document Registry & Query Controls)
with st.sidebar:
    st.markdown("### **DISPLAY SETTINGS**")
    theme_choice = st.radio(
        "Interface Theme",
        options=["Light Theme (Default)", "Dark Theme"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")

    st.markdown("### **CIVIC DOCUMENT REGISTRY**")
    uploaded_files = st.file_uploader(
        "Upload Official Scheme PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("INDEX UPLOADED DOCUMENTS", use_container_width=True):
            with st.spinner("Processing and indexing official documents..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                all_chunks = []
                doc_list = []

                for pdf_file in uploaded_files:
                    doc_list.append(pdf_file.name)
                    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                    for page_num in range(len(doc)):
                        text = doc[page_num].get_text("text").strip()
                        if text:
                            splits = splitter.split_text(text)
                            for s in splits:
                                all_chunks.append({
                                    "text": s,
                                    "source": pdf_file.name,
                                    "page": page_num + 1
                                })

                if all_chunks:
                    texts = [c["text"] for c in all_chunks]
                    embeddings = embedder.encode(texts, convert_to_numpy=True)
                    faiss.normalize_L2(embeddings)
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    index.add(embeddings)

                    tokenized_corpus = [t.lower().split() for t in texts]
                    bm25 = BM25Okapi(tokenized_corpus)

                    st.session_state.indexed_chunks = all_chunks
                    st.session_state.faiss_index = index
                    st.session_state.bm25_index = bm25
                    st.session_state.doc_names = doc_list
                    st.success(f"Indexed {len(all_chunks)} sections across {len(doc_list)} document(s).")

    st.markdown("---")
    st.markdown("### **QUERY CONTROLS**")
    response_lang = st.selectbox(
        "Response Language",
        options=["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"]
    )

    scope_options = ["All Indexed Documents"] + st.session_state.doc_names
    doc_scope = st.selectbox("Document Scope Filter", options=scope_options)

# 6. Dynamic Formal Government Color Palette
is_dark = (theme_choice == "Dark Theme")

if is_dark:
    bg_main = "#0F172A"          # Muted Deep Slate
    bg_sidebar = "#1E293B"       # Neutral Dark Slate
    bg_card = "#1E293B"
    text_main = "#F8FAFC"        # Crisp Off-White
    text_secondary = "#CBD5E1"
    border_col = "#334155"
    header_bg = "#1E293B"
    navy_accent = "#3B82F6"
    saffron_accent = "#D97706"
    btn_bg = "#1B365D"
else:
    bg_main = "#FFFFFF"          # High-Contrast Official White
    bg_sidebar = "#F3F4F6"       # Calm Light Neutral Gray
    bg_card = "#F8FAFC"
    text_main = "#111827"        # High-Contrast Charcoal Black
    text_secondary = "#374151"
    border_col = "#CBD5E1"
    header_bg = "#F8FAFC"
    navy_accent = "#1B365D"      # Official Government Ashoka Navy
    saffron_accent = "#D97706"   # Official Saffron Gold Accent
    btn_bg = "#1B365D"

emblem_src = get_permanent_emblem_src()

# 7. Inject Formal CSS & Permanent Central Government Header
st.markdown(f"""
<style>
    .stApp, [data-testid="stHeader"] {{
        background-color: {bg_main};
        color: {text_main};
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {bg_sidebar};
        border-right: 1px solid {border_col};
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {text_main} !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em;
    }}
    .stApp p, .stApp label, .stApp span, .stApp div {{
        color: {text_main};
    }}
    textarea, input, [data-baseweb="select"] > div {{
        background-color: {bg_card} !important;
        color: {text_main} !important;
        border: 1px solid {border_col} !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }}
    .stButton > button {{
        background-color: {btn_bg} !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid {navy_accent} !important;
        border-radius: 4px !important;
        padding: 0.55rem 1.3rem !important;
    }}
    .stButton > button p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    button[data-baseweb="tab"] p {{
        font-weight: 800 !important;
        font-size: 14px !important;
        letter-spacing: 0.03em;
    }}
    .central-govt-banner {{
        display: flex;
        align-items: center;
        gap: 22px;
        background-color: {header_bg};
        border: 1px solid {border_col};
        border-top: 5px solid {navy_accent};
        border-left: 6px solid {saffron_accent};
        border-radius: 4px;
        padding: 18px 24px;
        margin-bottom: 22px;
    }}
    .central-govt-banner img {{
        height: 76px;
        width: auto;
        object-fit: contain;
        background-color: #FFFFFF;
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px solid {border_col};
    }}
    .central-govt-text .ministry-tag {{
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {saffron_accent} !important;
        margin: 0 0 4px 0;
    }}
    .central-govt-text h1 {{
        font-size: 23px !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        color: {text_main} !important;
    }}
    .central-govt-text p {{
        font-size: 14px;
        font-weight: 500;
        margin: 0;
        color: {text_secondary} !important;
    }}
    .citation-box {{
        background-color: {bg_card};
        border: 1px solid {border_col};
        border-left: 4px solid {navy_accent};
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 12px;
    }}
</style>

<div class="central-govt-banner">
    <img src="{emblem_src}" alt="National Emblem of India" />
    <div class="central-govt-text">
        <div class="ministry-tag">Government of India &bull; Digital Citizen Information Service</div>
        <h1>Multilingual Civic Scheme &amp; Document Assistant</h1>
        <p>Citation-grounded public policy &amp; welfare scheme verification portal powered by Hybrid Retrieval (FAISS + BM25) and Gemini.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 8. Hybrid Retrieval & Gemini Answer Generation Function
def retrieve_and_answer(query: str, lang: str, scope: str):
    chunks = st.session_state.indexed_chunks
    if not chunks:
        return "No indexed documents found. Please upload and index official scheme PDFs in the left sidebar first.", []

    valid_indices = [
        i for i, c in enumerate(chunks)
        if scope == "All Indexed Documents" or c["source"] == scope
    ]
    if not valid_indices:
        return "No sections match the selected document scope.", []

    # Dense Retrieval (FAISS)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = st.session_state.faiss_index.search(q_emb, k=min(10, len(chunks)))
    dense_hits = {int(idx): float(sc) for idx, sc in zip(indices[0], scores[0]) if int(idx) in valid_indices}

    # Sparse Retrieval (BM25)
    bm25_scores = st.session_state.bm25_index.get_scores(query.lower().split())
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

    # Combine Hybrid Scores
    combined = []
    for idx in valid_indices:
        d_score = dense_hits.get(idx, 0.0)
        s_score = float(bm25_scores[idx]) / max_bm25
        hybrid_score = (0.65 * d_score) + (0.35 * s_score)
        combined.append((hybrid_score, chunks[idx]))

    combined.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [item[1] for item in combined[:4]]

    context_block = "\n\n".join(
        [f"[Document: {c['source']} | Page {c['page']}]\n{c['text']}" for c in top_chunks]
    )

    prompt = f"""You are an official public policy assistant. Answer the citizen's question strictly using ONLY the verified policy excerpts below.
Do not use emojis. Use clear, formal, easy-to-read language.
Cite the exact document name and page number inline (e.g., [Document.pdf, Page 3]).
If the answer is not present in the provided excerpts, state clearly: "The uploaded official documents do not contain information regarding this query."
Write the entire response in {lang}.

VERIFIED EXCERPTS:
{context_block}

CITIZEN QUESTION:
{query}
"""
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return "API key missing. Set GEMINI_API_KEY in environment variables or Streamlit secrets.", top_chunks

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=900,
        ),
    )
    return response.text, top_chunks

# 9. Formal Tabs (No Emojis, Bold Uppercase)
tab_ask, tab_audit = st.tabs(["ASK AND VERIFY", "QUERY AUDIT LOG"])

with tab_ask:
    st.markdown("**Enter your question about eligibility, benefits, required documents, or deadlines:**")
    user_query = st.text_area(
        "Citizen Query Input",
        placeholder="e.g., What is the annual financial assistance provided under the scheme, and what are the eligibility criteria?",
        height=115,
        label_visibility="collapsed"
    )

    if st.button("GENERATE VERIFIED ANSWER"):
        if not user_query.strip():
            st.warning("Please enter a valid question before submitting.")
        else:
            with st.spinner("Retrieving official citations and verifying response..."):
                answer, sources = retrieve_and_answer(user_query.strip(), response_lang, doc_scope)
                st.markdown("### **VERIFIED RESPONSE**")
                st.markdown(f"<div class='citation-box'>{answer}</div>", unsafe_allow_html=True)

                if sources:
                    st.markdown("### **SUPPORTING DOCUMENT CITATIONS**")
                    for idx, src in enumerate(sources, 1):
                        with st.expander(f"Citation {idx}: {src['source']} — Page {src['page']}"):
                            st.write(src["text"])

                st.session_state.audit_log.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "language": response_lang,
                    "scope": doc_scope,
                    "question": user_query.strip(),
                    "answer": answer
                })

with tab_audit:
    st.markdown("### **SESSION QUERY AUDIT LOG**")
    if not st.session_state.audit_log:
        st.info("No queries have been submitted in this session yet.")
    else:
        for entry in st.session_state.audit_log:
            with st.expander(f"[{entry['timestamp']}] ({entry['language']}) — {entry['question'][:80]}"):
                st.markdown(f"**Document Scope:** {entry['scope']}")
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Verified Answer:**\n\n{entry['answer']}")