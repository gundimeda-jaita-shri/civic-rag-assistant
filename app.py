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

UI_STRINGS = {
    "English": {
        "ministry": "Government of India • Digital Citizen Information Service",
        "title": "Multilingual Civic Scheme & Document Assistant",
        "subtitle": "Citation-grounded public policy & welfare scheme verification portal powered by Hybrid Retrieval (FAISS + BM25) and Gemini.",
        "display_settings": "DISPLAY SETTINGS",
        "light_theme": "Light Theme (Default)",
        "dark_theme": "Dark Theme",
        "doc_registry": "CIVIC DOCUMENT REGISTRY",
        "upload_label": "Upload Official Scheme PDFs",
        "index_btn": "INDEX UPLOADED DOCUMENTS",
        "query_controls": "QUERY CONTROLS",
        "resp_lang": "Response Language",
        "scope_filter": "Document Scope Filter",
        "all_docs": "All Indexed Documents",
        "tab_ask": "ASK AND VERIFY",
        "tab_audit": "QUERY AUDIT LOG",
        "prompt_label": "Enter your question about eligibility, benefits, required documents, or deadlines:",
        "placeholder": "e.g., What is the annual financial assistance provided under the scheme, and what are the eligibility criteria?",
        "gen_btn": "GENERATE VERIFIED ANSWER",
        "verified_resp": "VERIFIED RESPONSE",
        "citations": "SUPPORTING DOCUMENT CITATIONS",
        "audit_title": "SESSION QUERY AUDIT LOG",
        "no_queries": "No queries have been submitted in this session yet."
    },
    "Hindi": {
        "ministry": "भारत सरकार • डिजिटल नागरिक सूचना सेवा",
        "title": "बहुभाषी नागरिक योजना एवं दस्तावेज़ सहायक",
        "subtitle": "हाइब्रिड रिट्रीवल (FAISS + BM25) और जेमिनी द्वारा संचालित प्रमाण-आधारित सार्वजनिक नीति एवं कल्याण योजना सत्यापन पोर्टल।",
        "display_settings": "प्रदर्शन सेटिंग्स",
        "light_theme": "लाइट थीम (डिफ़ॉल्ट)",
        "dark_theme": "डार्क थीम",
        "doc_registry": "नागरिक दस्तावेज़ रजिस्ट्री",
        "upload_label": "आधिकारिक योजना PDF अपलोड करें",
        "index_btn": "अपलोड किए गए दस्तावेज़ इंडेक्स करें",
        "query_controls": "प्रश्न नियंत्रण",
        "resp_lang": "उत्तर की भाषा",
        "scope_filter": "दस्तावेज़ फ़िल्टर",
        "all_docs": "सभी इंडेक्स किए गए दस्तावेज़",
        "tab_ask": "पूछें और सत्यापित करें",
        "tab_audit": "प्रश्न ऑडिट लॉग",
        "prompt_label": "पात्रता, लाभ, आवश्यक दस्तावेज़ या अंतिम तिथि के बारे में अपना प्रश्न दर्ज करें:",
        "placeholder": "उदाहरण: योजना के अंतर्गत वार्षिक वित्तीय सहायता कितनी है और पात्रता मानदंड क्या हैं?",
        "gen_btn": "सत्यापित उत्तर प्राप्त करें",
        "verified_resp": "सत्यापित उत्तर",
        "citations": "सहायक दस्तावेज़ उद्धरण",
        "audit_title": "सत्र प्रश्न ऑडिट लॉग",
        "no_queries": "इस सत्र में अभी तक कोई प्रश्न दर्ज नहीं किया गया है।"
    },
    "Telugu": {
        "ministry": "భారత ప్రభుత్వం • డిజిటల్ పౌర సమాచార సేవ",
        "title": "బహుభాషా పౌర పథకం & పత్రాల సహాయకుడు",
        "subtitle": "హైబ్రిడ్ రిట్రీవల్ (FAISS + BM25) మరియు జెమిని ద్వారా ఆధారితమైన ప్రజా విధాన & సంక్షేమ పథకాల ధృవీకరణ పోర్టల్.",
        "display_settings": "డిస్‌ప్లే సెట్టింగ్‌లు",
        "light_theme": "లైట్ థీమ్ (డిఫాల్ట్)",
        "dark_theme": "డార్క్ థీమ్",
        "doc_registry": "పౌర పత్రాల రిజిస్ట్రీ",
        "upload_label": "అధికారిక పథకం PDFలను అప్‌లోడ్ చేయండి",
        "index_btn": "అప్‌లోడ్ చేసిన పత్రాలను ఇండెక్స్ చేయండి",
        "query_controls": "ప్రశ్న నియంత్రణలు",
        "resp_lang": "సమాధాన భాష",
        "scope_filter": "పత్రం ఫిల్టర్",
        "all_docs": "అన్ని ఇండెక్స్ చేసిన పత్రాలు",
        "tab_ask": "అడగండి మరియు ధృవీకరించండి",
        "tab_audit": "ప్రశ్నల ఆడిట్ లాగ్",
        "prompt_label": "అర్హత, ప్రయోజనాలు, అవసరమైన పత్రాలు లేదా గడువు తేదీల గురించి మీ ప్రశ్నను నమోదు చేయండి:",
        "placeholder": "ఉదాహరణ: పథకం కింద అందించే వార్షిక ఆర్థిక సహాయం ఎంత మరియు అర్హత ప్రమాణాలు ఏమిటి?",
        "gen_btn": "ధృవీకరించబడిన సమాధానం పొందండి",
        "verified_resp": "ధృవీకరించబడిన సమాధానం",
        "citations": "సహాయక పత్రాల ఆధారాలు",
        "audit_title": "సెషన్ ప్రశ్నల ఆడిట్ లాగ్",
        "no_queries": "ఈ సెషన్‌లో ఇంకా ఎటువంటి ప్రశ్నలు సమర్పించబడలేదు."
    },
    "Tamil": {
        "ministry": "இந்திய அரசு • டிஜிட்டல் குடிமக்கள் தகவல் சேவை",
        "title": "பன்மொழி குடிமக்கள் திட்டம் & ஆவண உதவியாளர்",
        "subtitle": "ஹைப்ரிட் மீட்டெடுப்பு (FAISS + BM25) மற்றும் ஜெமினி மூலம் இயக்கப்படும் பொதுக் கொள்கை மற்றும் நலத்திட்ட சரிபார்ப்பு தளம்.",
        "display_settings": "காட்சி அமைப்புகள்",
        "light_theme": "ஒளி தீம் (இயல்புநிலை)",
        "dark_theme": "இருண்ட தீம்",
        "doc_registry": "குடிமக்கள் ஆவணப் பதிவேடு",
        "upload_label": "அதிகாரப்பூர்வ திட்ட PDFகளை பதிவேற்றவும்",
        "index_btn": "பதிவேற்றிய ஆவணங்களை அட்டவணைப்படுத்து",
        "query_controls": "கேள்வி கட்டுப்பாடுகள்",
        "resp_lang": "பதில் மொழி",
        "scope_filter": "ஆவண வடிகட்டி",
        "all_docs": "அனைத்து ஆவணங்களும்",
        "tab_ask": "கேட்டு சரிபார்க்கவும்",
        "tab_audit": "கேள்வி தணிக்கை பதிவு",
        "prompt_label": "தகுதி, சலுகைகள், தேவையான ஆவணங்கள் அல்லது காலக்கெடு பற்றிய உங்கள் கேள்வியை உள்ளிடவும்:",
        "placeholder": "எ.கா., திட்டத்தின் கீழ் வழங்கப்படும் ஆண்டு நிதியுதவி என்ன மற்றும் தகுதி வரம்புகள் யாவை?",
        "gen_btn": "சரிபார்க்கப்பட்ட பதிலைப் பெறுக",
        "verified_resp": "சரிபார்க்கப்பட்ட பதில்",
        "citations": "ஆதார ஆவண மேற்கோள்கள்",
        "audit_title": "அமர்வு கேள்வி தணிக்கை பதிவு",
        "no_queries": "இந்த அமர்வில் இதுவரை எந்த கேள்வியும் சமர்ப்பிக்கப்படவில்லை."
    },
    "Marathi": {
        "ministry": "भारत सरकार • डिजिटल नागरिक माहिती सेवा",
        "title": "बहुभाषिक नागरी योजना आणि दस्तऐवज सहाय्यक",
        "subtitle": "हायब्रीड रिट्रीव्हल (FAISS + BM25) आणि जेमिनी द्वारे संचालित सार्वजनिक धोरण आणि कल्याणकारी योजना पडताळणी पोर्टल.",
        "display_settings": "प्रदर्शन सेटिंग्ज",
        "light_theme": "लाइट थीम (डीफॉल्ट)",
        "dark_theme": "डार्क थीम",
        "doc_registry": "नागरी दस्तऐवज नोंदणी",
        "upload_label": "अधिकृत योजना PDF अपलोड करा",
        "index_btn": "अपलोड केलेले दस्तऐवज इंडेक्स करा",
        "query_controls": "प्रश्न नियंत्रणे",
        "resp_lang": "उत्तराची भाषा",
        "scope_filter": "दस्तऐवज फिल्टर",
        "all_docs": "सर्व इंडेक्स केलेले दस्तऐवज",
        "tab_ask": "विचारा आणि पडताळणी करा",
        "tab_audit": "प्रश्न ऑडिट लॉग",
        "prompt_label": "पात्रता, लाभ, आवश्यक कागदपत्रे किंवा अंतिम मुदतीबद्दल तुमचा प्रश्न प्रविष्ट करा:",
        "placeholder": "उदा., योजनेअंतर्गत वार्षिक आर्थिक मदत किती आहे आणि पात्रता निकष काय आहेत?",
        "gen_btn": "पडताळलेले उत्तर मिळवा",
        "verified_resp": "पडताळलेले उत्तर",
        "citations": "सहाय्यक दस्तऐवज संदर्भ",
        "audit_title": "सत्र प्रश्न ऑडिट लॉग",
        "no_queries": "या सत्रात अद्याप कोणतेही प्रश्न सबमिट केलेले नाहीत."
    },
    "Bengali": {
        "ministry": "ভারত সরকার • ডিজিটাল নাগরিক তথ্য পরিষেবা",
        "title": "বহুভাষিক নাগরিক প্রকল্প ও নথি সহায়ক",
        "subtitle": "হাইব্রিড রিট্রিভাল (FAISS + BM25) এবং জেমিনি দ্বারা চালিত জননীতি ও কল্যাণ প্রকল্প যাচাইকরণ পোর্টাল।",
        "display_settings": "ডিসপ্লে সেটিংস",
        "light_theme": "লাইট থিম (ডিফল্ট)",
        "dark_theme": "ডার্ক থিম",
        "doc_registry": "নাগরিক নথি নিবন্ধন",
        "upload_label": "সরকারি প্রকল্পের PDF আপলোড করুন",
        "index_btn": "আপলোড করা নথি ইনডেক্স করুন",
        "query_controls": "প্রশ্ন নিয়ন্ত্রণ",
        "resp_lang": "উত্তরের ভাষা",
        "scope_filter": "নথি ফিল্টার",
        "all_docs": "সমস্ত ইনডেক্স করা নথি",
        "tab_ask": "জিজ্ঞাসা ও যাচাই করুন",
        "tab_audit": "প্রশ্ন অডিট লগ",
        "prompt_label": "যোগ্যতা, সুবিধা, প্রয়োজনীয় নথি বা সময়সীমা সম্পর্কে আপনার প্রশ্ন লিখুন:",
        "placeholder": "যেমন: প্রকল্পের অধীনে বার্ষিক আর্থিক সহায়তা কত এবং যোগ্যতার মানদণ্ড কী?",
        "gen_btn": "যাচাইকৃত উত্তর তৈরি করুন",
        "verified_resp": "যাচাইকৃত উত্তর",
        "citations": "সহায়ক নথির উদ্ধৃতি",
        "audit_title": "সেশন প্রশ্ন অডিট লগ",
        "no_queries": "এই সেশনে এখনও কোনো প্রশ্ন জমা দেওয়া হয়নি।"
    }
}

# 2. Permanent Local Emblem Loader (Supports .in_, .in_*, .png, .jpg, .webp, .svg)
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
        os.path.join(base_dir, "central_govt*"),
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

# 3. Initialize Session State (Including Language State for Instant UI Switching)
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
if "lang_select" not in st.session_state:
    st.session_state.lang_select = "English"

response_lang = st.session_state.lang_select
t = UI_STRINGS.get(response_lang, UI_STRINGS["English"])

# 4. Load Multilingual Embedding Model (Cached)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embedder = load_embedder()

# 5. Sidebar Controls (Language Selector First, Then Localized Controls)
with st.sidebar:
    st.markdown(f"### **{t['query_controls']}**")
    st.selectbox(
        t["resp_lang"],
        options=list(UI_STRINGS.keys()),
        key="lang_select"
    )
    response_lang = st.session_state.lang_select
    t = UI_STRINGS.get(response_lang, UI_STRINGS["English"])

    scope_options = [t["all_docs"]] + st.session_state.doc_names
    doc_scope = st.selectbox(t["scope_filter"], options=scope_options)
    st.markdown("---")

    st.markdown(f"### **{t['display_settings']}**")
    theme_choice = st.radio(
        "Interface Theme",
        options=[t["light_theme"], t["dark_theme"]],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")

    st.markdown(f"### **{t['doc_registry']}**")
    uploaded_files = st.file_uploader(
        t["upload_label"],
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button(t["index_btn"], use_container_width=True):
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

                    tokenized_corpus = [txt.lower().split() for txt in texts]
                    bm25 = BM25Okapi(tokenized_corpus)

                    st.session_state.indexed_chunks = all_chunks
                    st.session_state.faiss_index = index
                    st.session_state.bm25_index = bm25
                    st.session_state.doc_names = doc_list
                    st.success(f"Indexed {len(all_chunks)} sections across {len(doc_list)} document(s).")

# 6. Dynamic Formal Government Color Palette
is_dark = (theme_choice == t["dark_theme"])

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
    .stApp p, .stApp label, .stApp li {{
        color: {text_main} !important;
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
        color: {text_main} !important;
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
</style>

<div class="central-govt-banner">
    <img src="{emblem_src}" alt="National Emblem of India" />
    <div class="central-govt-text">
        <div class="ministry-tag">{t['ministry']}</div>
        <h1>{t['title']}</h1>
        <p>{t['subtitle']}</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 8. Hybrid Retrieval & Gemini Answer Generation Function
def retrieve_and_answer(query: str, lang: str, scope: str, all_docs_label: str):
    chunks = st.session_state.indexed_chunks
    if not chunks:
        return "No indexed documents found. Please upload and index official scheme PDFs in the left sidebar first.", []

    valid_indices = [
        i for i, c in enumerate(chunks)
        if scope == all_docs_label or c["source"] == scope
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
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        return "API key missing. Set GEMINI_API_KEY in environment variables or Streamlit secrets.", top_chunks

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=900,
            ),
        )
        return response.text or "Model returned an empty response.", top_chunks
    except Exception as e:
        return f"Gemini API Error: {str(e)}", top_chunks

# 9. Formal Tabs (Localized, No Emojis, Bold Uppercase)
tab_ask, tab_audit = st.tabs([t["tab_ask"], t["tab_audit"]])

with tab_ask:
    st.markdown(f"**{t['prompt_label']}**")
    user_query = st.text_area(
        "Citizen Query Input",
        placeholder=t["placeholder"],
        height=115,
        label_visibility="collapsed"
    )

    if st.button(t["gen_btn"]):
        if not user_query.strip():
            st.warning("Please enter a valid question before submitting.")
        else:
            with st.spinner("Retrieving official citations and verifying response..."):
                answer, sources = retrieve_and_answer(user_query.strip(), response_lang, doc_scope, t["all_docs"])
                st.markdown(f"### **{t['verified_resp']}**")
                with st.container(border=True):
                    st.markdown(answer)

                if sources:
                    st.markdown(f"### **{t['citations']}**")
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
    st.markdown(f"### **{t['audit_title']}**")
    if not st.session_state.audit_log:
        st.info(t["no_queries"])
    else:
        for entry in st.session_state.audit_log:
            with st.expander(f"[{entry['timestamp']}] ({entry['language']}) — {entry['question'][:80]}"):
                st.markdown(f"**{t['scope_filter']}:** {entry['scope']}")
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**{t['verified_resp']}:**")
                st.markdown(entry["answer"])