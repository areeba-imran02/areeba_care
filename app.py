"""
Healthcare Assistant
--------------------
A grounded, retrieval-augmented healthcare information assistant built on top
of a local PDF knowledge base. This single-file Streamlit application covers
the full pipeline:

    PDFs -> text extraction -> cleaning -> chunking -> embeddings ->
    FAISS index -> retrieval -> grounded Groq answer -> gTTS audio

Educational / demonstration project only. This is NOT a real medical
diagnosis or treatment system.
"""

import os
import re
import io
import glob
import pickle
import hashlib
import tempfile
import time

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL_SIZE = "small"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL_NAME = "openai/gpt-oss-20b"

CHUNK_SIZE_WORDS = 600
CHUNK_OVERLAP_WORDS = 100
TOP_K = 3
RELEVANCE_THRESHOLD = 0.30
MAX_ANSWER_TOKENS = 500
HISTORY_TURNS_SENT = 4

LANGUAGE_OPTIONS = ["English", "Urdu", "Roman Urdu"]

LANGUAGE_INSTRUCTIONS = {
    "English": "Respond entirely in clear, professional English.",
    "Urdu": "Respond entirely in the Urdu language, written using Urdu script.",
    "Roman Urdu": "Respond entirely in Roman Urdu (the Urdu language written using English/Latin letters, not Urdu script).",
}

GTTS_LANG_MAP = {
    "English": "en",
    "Urdu": "ur",
    "Roman Urdu": "ur",
}

WHISPER_LANG_HINT = {
    "English": "en",
    "Urdu": "ur",
    "Roman Urdu": "ur",
}

QUICK_PROMPT_OPTIONS = [
    "-- Select a sample question to ask --",
    "🚨 What are the emergency protocols and helpline numbers?",
    "🕒 What are the visiting hours for general wards and ICU?",
    "🏥 Which specialized medical departments and consultants are available?",
    "📋 What documents and steps are required for patient admission?",
    "💳 What insurance panels and billing policies are supported?",
    "🩺 How can I book an appointment with a specialist doctor?",
    "🧪 What are the operating timings for laboratory and radiology services?",
    "📜 What is the step-by-step procedure for patient discharge?",
    "💊 Is the hospital pharmacy open 24/7 for medicine delivery?",
    "🚙 What are the parking and ambulance service facilities available?"
]

NOT_FOUND_MESSAGE = (
    "I couldn't find enough relevant information in the hospital "
    "knowledge base to answer that question accurately."
)

KB_NOT_READY_MESSAGE = (
    "The knowledge base is not ready yet. Please add PDF "
    "documents to the knowledge_base folder and use \"Rebuild Index\"."
)

MISSING_KEY_MESSAGE = (
    "Groq API key is not configured. Please set the GROQ_API_KEY "
    "environment variable before using the assistant."
)

GROQ_ERROR_MESSAGE = (
    "Healthcare Assistant is temporarily unable to generate an answer. Please wait a "
    "moment and try again."
)

SYSTEM_PROMPT_TEMPLATE = """You are the Healthcare Assistant, an educational information \
assistant for a hospital system.

STRICT RULES YOU MUST ALWAYS FOLLOW:
1. Answer ONLY using the knowledge base context provided below. Never use outside or general medical knowledge.
2. Never invent, guess, or assume information that is not explicitly present in the provided context.
3. If the provided context does not contain enough information to answer the question, clearly say so.
4. Never diagnose a medical condition.
5. Never prescribe medication or dosages.
6. Never recommend a specific treatment.
7. Never claim to replace a licensed doctor or medical professional.
8. Keep answers clear, concise, and genuinely useful.
9. {language_instruction}
10. Remember that this is an educational hospital knowledge base used for demonstration purposes only.

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

# ---------------------------------------------------------------------------
# Professional Clinical UI — design tokens & global styling
#
# Palette:
#   Deep teal-navy   (#0c2f35 / #12474b) -> sidebar, hero, header strips
#   Clinical teal    (#1c7d74)           -> primary actions, active/chat accent
#   Warm amber       (#b8823a)           -> secondary accent, reference/FAQ panel
#   Neutral surfaces (#eef2f2 / #ffffff) -> page background / card surfaces
#   Ink              (#16262b / #55696c) -> body text / muted text (both pass
#                                           WCAG AA contrast on white and on
#                                           the deep navy panels)
#
# Every card in the layout (hero, chat panel, FAQ panel, hospital-info card,
# footer) gets an explicit, visible border on all four sides, either via a
# real `border:` rule or via Streamlit's native bordered container, so
# sections read as distinct, separated regions rather than floating text.
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg: #eef2f2;
            --surface: #ffffff;
            --surface-tint: #f4f9f8;

            --border: #d3dfdf;
            --border-strong: #aec3c4;

            --ink: #16262b;
            --ink-muted: #55696c;

            --navy-900: #0c2f35;
            --navy-700: #12474b;

            --teal-600: #1c7d74;
            --teal-100: #e2f2ef;

            --amber-600: #b8823a;
            --amber-100: #faf1e2;

            --radius-lg: 18px;
            --radius-md: 12px;
        }

        html, body, .stApp {
            background-color: var(--bg) !important;
            font-family: 'Inter', sans-serif;
            color: var(--ink);
        }

        h1, h2, h3, h4, .hero-banner h1, .chatbot-title, .panel-title {
            font-family: 'Manrope', sans-serif;
        }

        /* ---------------------------------------------------------------
           Hero banner — deliberately its own palette (deep teal-emerald
           blend) so it reads as a distinct, premium "masthead" rather
           than matching the plain navy of the chat panel title bar.
           A soft glowing teal ring + top accent line gives it a
           polished, unique identity at a glance.
        --------------------------------------------------------------- */
        .hero-banner {
            position: relative;
            background: linear-gradient(120deg, #0a2e33 0%, #124a49 45%, #1c7d74 100%);
            border: 1px solid rgba(111, 214, 200, 0.35);
            border-radius: var(--radius-lg);
            padding: 1.9rem 2.2rem;
            margin-bottom: 1.4rem;
            overflow: hidden;
            box-shadow:
                0 14px 34px -16px rgba(12, 47, 53, 0.65),
                0 0 0 1px rgba(111, 214, 200, 0.06) inset;
        }
        .hero-banner::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 88% 12%, rgba(111, 214, 200, 0.28), transparent 55%);
            pointer-events: none;
        }
        .hero-banner::after {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6fd6c8 0%, #1c7d74 50%, #b8823a 100%);
        }
        .hero-top-row {
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 0.65rem;
        }
        .hero-banner h1 {
            color: #ffffff !important;
            margin: 0;
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.015em;
            text-shadow: 0 2px 18px rgba(0, 0, 0, 0.25);
        }
        .hero-banner .hero-welcome {
            position: relative;
            color: #eafaf7 !important;
            margin: 0 0 0.35rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .hero-banner p.hero-sub {
            position: relative;
            color: #cfe9e5 !important;
            margin: 0;
            font-size: 0.97rem;
            line-height: 1.6;
            max-width: 780px;
            font-weight: 400;
        }
        .creator-badge {
            position: relative;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.35);
            color: #eafaf7 !important;
            padding: 0.4rem 0.9rem;
            border-radius: 30px;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
            backdrop-filter: blur(2px);
        }

        /* ---------------------------------------------------------------
           Sidebar
        --------------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            background-color: var(--navy-900) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        section[data-testid="stSidebar"] * {
            color: #eafaf7 !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: 'Manrope', sans-serif;
            font-weight: 700;
        }

        .sidebar-brand {
            padding-bottom: 0.9rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.14);
        }
        .sidebar-brand h2 {
            font-size: 1.35rem;
            margin: 0;
        }
        .sidebar-section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin: 0 0 0.5rem 0;
        }
        .sidebar-section-label.teal { color: #6fd6c8 !important; }
        .sidebar-section-label.amber { color: #e7be82 !important; }

        .sidebar-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: var(--radius-md);
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .sidebar-card ul {
            margin: 0;
            padding-left: 1.1rem;
            font-size: 0.86rem;
            line-height: 1.55;
        }

        /* ==================================================================
           SELECTBOX VISIBILITY — FULL OVERRIDE
           Covers: (a) the closed/collapsed selectbox box wherever it sits
           (sidebar or main area), and (b) the open dropdown options list,
           which BaseWeb renders in a body-level portal outside the sidebar
           DOM, so sidebar-scoped rules never reach it on their own.
           Every rule below is solid-background + solid-text, no
           transparency, so it can never blend into the dark sidebar or
           inherit an invisible color from a parent.
        ================================================================== */

        /* Closed selectbox control, anywhere in the app */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1.5px solid var(--teal-600) !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="select"] > div * {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }
        /* SIDEBAR OVERRIDE — targets Streamlit's own stSelectbox wrapper
           together with the BaseWeb select, which is MORE specific than
           the blanket `section[data-testid="stSidebar"] * { color:
           #eafaf7 }` rule above (two attribute selectors instead of one),
           so it wins regardless of source order. This is what was making
           "English" render as near-invisible faded text before. */
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
            background-color: transparent !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {
            background-color: #ffffff !important;
            border: 1.5px solid var(--teal-600) !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] * {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] svg {
            fill: var(--ink) !important;
        }

        /* Open dropdown options list (rendered in a body-level portal) */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {
            z-index: 999999 !important;
        }
        div[data-baseweb="popover"] [data-baseweb="menu"],
        div[data-baseweb="popover"] ul[role="listbox"],
        div[data-baseweb="popover"] div[role="listbox"] {
            background-color: #ffffff !important;
            border: 1.5px solid var(--border-strong) !important;
            border-radius: 10px !important;
            box-shadow: 0 10px 24px -10px rgba(12, 47, 53, 0.35) !important;
        }
        div[data-baseweb="popover"] li[role="option"],
        div[data-baseweb="popover"] li[role="option"] *,
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [role="option"] * {
            background-color: #ffffff !important;
            color: var(--ink) !important;
        }
        div[data-baseweb="popover"] li[role="option"]:hover,
        div[data-baseweb="popover"] li[aria-selected="true"],
        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] [aria-selected="true"] {
            background-color: var(--teal-100) !important;
            color: var(--ink) !important;
        }
        /* ================================================================== */

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--teal-600) 0%, #16645d 100%) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.2rem !important;
            box-shadow: 0 4px 14px rgba(28, 125, 116, 0.25) !important;
            transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(28, 125, 116, 0.4) !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            color: #ffffff !important;
            width: 100%;
            box-shadow: none !important;
        }

        /* ---------------------------------------------------------------
           Bordered panels (Streamlit's native container border)
           Used for: the chat panel and the FAQ panel. A visible 1.5px
           border on all four sides, distinct background, and rounded
           corners so each region reads as its own enclosed card.
        --------------------------------------------------------------- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1.5px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            background: var(--surface) !important;
            box-shadow: 0 6px 20px -14px rgba(12, 47, 53, 0.3);
        }

        /* Panel title bars (rendered as the first element inside a
           bordered container) */
        .panel-title-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0.85rem 1.1rem;
            border-radius: var(--radius-md);
            margin-bottom: 1rem;
        }
        .panel-title-bar.chat {
            background: linear-gradient(135deg, var(--navy-900) 0%, var(--navy-700) 100%);
        }
        .panel-title-bar.faq {
            background: linear-gradient(135deg, #8f611f 0%, var(--amber-600) 100%);
        }
        .panel-title-bar .panel-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.16);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }
        .panel-title-bar .panel-title {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #ffffff;
        }
        .panel-title-bar .panel-subtitle {
            margin: 0;
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.85);
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            border-radius: var(--radius-md) !important;
            padding: 0.95rem 1.05rem !important;
            margin-bottom: 0.85rem !important;
            border: 1px solid var(--border) !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background-color: var(--teal-100) !important;
            border-color: #bfe3dc !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background-color: var(--surface-tint) !important;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span {
            color: var(--ink) !important;
        }

        .response-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.76rem;
            color: var(--navy-700) !important;
            font-weight: 600;
            margin-top: 0.6rem;
            padding-top: 0.5rem;
            border-top: 1px dashed var(--border-strong);
        }

        /* Text input inside the chat form */
        .stTextInput input {
            color: var(--ink) !important;
            background-color: #ffffff !important;
            border-radius: 10px !important;
        }

        /* FAQ helper caption */
        .faq-caption {
            font-size: 0.83rem;
            color: var(--ink-muted);
            margin-bottom: 0.7rem;
        }

        /* Footer */
        .custom-footer {
            background: linear-gradient(135deg, var(--navy-900) 0%, var(--navy-700) 100%);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: var(--radius-lg);
            padding: 1.1rem;
            text-align: center;
            margin-top: 1.6rem;
            box-shadow: 0 10px 28px -18px rgba(12, 47, 53, 0.55);
        }
        .custom-footer h3 { color: #ffffff; margin: 0; font-size: 1.05rem; }
        .custom-footer p { color: #d7e7e5; margin: 0.25rem 0; }
        .custom-footer p.fine-print { font-size: 0.76rem; opacity: 0.85; }

        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached model / client loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_whisper_model():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key):
    if not api_key:
        return None
    from groq import Groq  # let ImportError surface if the package isn't installed
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# PDF processing pipeline
# ---------------------------------------------------------------------------
def load_pdf_files():
    return sorted(glob.glob(os.path.join(KB_DIR, "*.pdf")))


def extract_pdf_text(path):
    try:
        import fitz  # PyMuPDF
        text_parts = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts), None
    except Exception as e:
        return None, str(e)


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def chunk_text(text, source_name, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append({"text": chunk_str, "source": source_name})
        if end == len(words):
            break
        start += step
    return chunks


def build_embeddings(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings):
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, chunks):
    import faiss
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_faiss_index_from_disk():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        return None, None
    try:
        import faiss
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except Exception:
        return None, None


def load_or_build_knowledge_base(force_rebuild=False):
    if not force_rebuild:
        index, chunks = load_faiss_index_from_disk()
        if index is not None and chunks:
            st.session_state.kb_index = index
            st.session_state.kb_chunks = chunks
            st.session_state.kb_loaded_count = len({c["source"] for c in chunks})
            st.session_state.kb_failed = []
            st.session_state.kb_status = "ready"
            return

    pdf_paths = load_pdf_files()
    if not pdf_paths:
        st.session_state.kb_index = None
        st.session_state.kb_chunks = None
        st.session_state.kb_loaded_count = 0
        st.session_state.kb_failed = []
        st.session_state.kb_status = "no_pdfs"
        return

    try:
        model = get_embedding_model()
    except Exception:
        st.session_state.kb_status = "embedding_error"
        return

    all_chunks = []
    failed_files = []
    for path in pdf_paths:
        fname = os.path.basename(path)
        text, err = extract_pdf_text(path)
        if err or text is None:
            failed_files.append(fname)
            continue
        cleaned = clean_text(text)
        if not cleaned:
            failed_files.append(fname)
            continue
        all_chunks.extend(chunk_text(cleaned, fname))

    if not all_chunks:
        st.session_state.kb_index = None
        st.session_state.kb_chunks = None
        st.session_state.kb_loaded_count = 0
        st.session_state.kb_failed = failed_files
        st.session_state.kb_status = "no_pdfs"
        return

    try:
        embeddings = build_embeddings(all_chunks, model)
        index = build_faiss_index(embeddings)
        save_faiss_index(index, all_chunks)
    except Exception:
        st.session_state.kb_status = "faiss_error"
        return

    st.session_state.kb_index = index
    st.session_state.kb_chunks = all_chunks
    st.session_state.kb_loaded_count = len({c["source"] for c in all_chunks})
    st.session_state.kb_failed = failed_files
    st.session_state.kb_status = "ready"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_context(query, index, chunks, model, top_k=TOP_K, threshold=RELEVANCE_THRESHOLD):
    if index is None or not chunks:
        return []
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        if score < threshold:
            continue
        c = chunks[idx]
        results.append({"text": c["text"], "source": c["source"], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# Answer generation (Groq)
# ---------------------------------------------------------------------------
def _call_groq(client, messages, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=MAX_ANSWER_TOKENS,
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        import traceback
        status_code = getattr(e, "status_code", None)
        err_type = type(e).__name__
        if err_type == "RateLimitError" or status_code == 429:
            return None, f"rate_limit::{e}"
        if err_type == "NotFoundError" or status_code == 404:
            return None, f"model_unavailable::{e}"
        return None, f"{err_type}: {e}\n{traceback.format_exc()}"


def generate_answer(query, context_chunks, language, history):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None, "missing_key"

    try:
        client = get_groq_client(api_key)
    except Exception as e:
        return None, f"client_init_error: {e!r}"

    if client is None:
        return None, "client_init_error: get_groq_client() returned None (groq package likely not installed, or client creation raised and was swallowed)"

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        language_instruction=LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["English"]),
        context=context_text,
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-HISTORY_TURNS_SENT:]:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": query})

    answer, error = _call_groq(client, messages, GROQ_MODEL_NAME)

    if error and (error.startswith("rate_limit::") or error.startswith("model_unavailable::")) and GROQ_FALLBACK_MODEL_NAME:
        # Primary model's daily quota is exhausted, or it's no longer
        # available on this account/tier — the fallback model has its own
        # separate quota and availability, so try it before giving up.
        reason = "rate-limited" if error.startswith("rate_limit::") else "unavailable"
        print(f"[Groq] {GROQ_MODEL_NAME} {reason}, falling back to {GROQ_FALLBACK_MODEL_NAME}")
        fallback_answer, fallback_error = _call_groq(client, messages, GROQ_FALLBACK_MODEL_NAME)
        if fallback_answer:
            return fallback_answer, None
        return None, f"{error} | fallback ({GROQ_FALLBACK_MODEL_NAME}) also failed: {fallback_error}"

    return answer, error


# ---------------------------------------------------------------------------
# Voice transcription
# ---------------------------------------------------------------------------
def transcribe_audio(audio_bytes, language):
    model = get_whisper_model()
    if model is None:
        return None, "Speech recognition is currently unavailable."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        hint = WHISPER_LANG_HINT.get(language)
        segments, _info = model.transcribe(tmp_path, language=hint, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            return None, "No speech was detected in the recording."
        return text, None
    except Exception:
        return None, "The voice recording could not be transcribed. Please try again."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Text-to-speech (gTTS)
# ---------------------------------------------------------------------------
def generate_tts(text, language):
    if not text:
        return None
    try:
        from gtts import gTTS
        lang_code = GTTS_LANG_MAP.get(language, "en")
        tts = gTTS(text=text, lang=lang_code)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sources rendering
# ---------------------------------------------------------------------------
def render_sources(context_chunks):
    if not context_chunks:
        return
    with st.expander("📚 Knowledge Base Sources"):
        for c in context_chunks:
            st.markdown(f"**📄 Document:** `{c['source']}`")
            st.caption(f"Relevance Score: {c['score']:.2f}")
            with st.expander("Show extracted segment text", expanded=False):
                st.write(c["text"])


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "chat_history": [],
        "language": "English",
        "kb_index": None,
        "kb_chunks": None,
        "kb_loaded_count": 0,
        "kb_failed": [],
        "kb_status": "not_loaded",
        "last_audio_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Smart Greetings Check
# ---------------------------------------------------------------------------
def is_greeting(query):
    cleaned = re.sub(r"[^\w\s]", "", query.lower().strip())
    greetings = {"hi", "hello", "hey", "salam", "aoa", "assalam o alaikum", "greetings"}
    return cleaned in greetings


def get_greeting_response(language):
    if language == "Urdu":
        return "السلام علیکم! میں آپ کا Healthcare Assistant ہوں۔ میں آپ کی کیا مدد کر سکتا ہوں؟ آپ ہسپتال کی سہولیات، رجسٹریشن اور دیگر معلومات کے بارے میں پوچھ سکتے ہیں۔"
    elif language == "Roman Urdu":
        return "Aoa! Main aap ka Healthcare Assistant hoon. Main aap ki kya madad kar sakta hoon? Aap hospital ki visiting hours, registration ya kisi bhi policy ke baray mein pooch saktay hain."
    else:
        return "Hello! Welcome to Healthcare Assistant. How can I assist you today? You can ask me about hospital services, visiting hours, registration, and patient guidelines."


def get_rate_limit_message(language, error_text):
    wait_str = None
    match = re.search(r"try again in ([\d.]+m[\d.]+s|[\d.]+s|[\d.]+m)", error_text or "")
    if match:
        wait_str = match.group(1)

    if language == "Urdu":
        if wait_str:
            return f"معذرت، آج کے لیے AI ماڈل کی استعمال کی حد (rate limit) مکمل ہو چکی ہے۔ براہِ کرم تقریباً {wait_str} بعد دوبارہ کوشش کریں۔"
        return "معذرت، آج کے لیے AI ماڈل کی استعمال کی حد مکمل ہو چکی ہے۔ براہِ کرم کچھ دیر بعد دوبارہ کوشش کریں۔"
    elif language == "Roman Urdu":
        if wait_str:
            return f"Maazrat, aaj ke liye AI model ki usage limit (rate limit) poori ho chuki hai. Meharbani kar ke taqreeban {wait_str} baad dobara koshish karein."
        return "Maazrat, aaj ke liye AI model ki usage limit poori ho chuki hai. Meharbani kar ke thodi der baad dobara koshish karein."
    else:
        if wait_str:
            return f"We've hit today's usage limit for the AI model. Please try again in about {wait_str}."
        return "We've hit today's usage limit for the AI model. Please try again in a little while."


# ---------------------------------------------------------------------------
# Core question handling
# ---------------------------------------------------------------------------
def handle_question(query, label=None):
    query = (query or "").strip()
    if not query:
        return

    start_time = time.time()
    display_text = query if not label else f"{label}\n\n{query}"
    st.session_state.chat_history.append({"role": "user", "content": display_text})

    language = st.session_state.language

    # Quick Greeting Handling
    if is_greeting(query):
        answer = get_greeting_response(language)
        context_chunks = []
    else:
        with st.spinner("Analyzing knowledge base & generating response..."):
            if st.session_state.kb_status != "ready" or st.session_state.kb_index is None:
                answer = KB_NOT_READY_MESSAGE
                context_chunks = []
            else:
                model = get_embedding_model()
                context_chunks = retrieve_context(
                    query, st.session_state.kb_index, st.session_state.kb_chunks, model
                )
                if not context_chunks:
                    answer = NOT_FOUND_MESSAGE
                else:
                    answer, error = generate_answer(query, context_chunks, language, st.session_state.chat_history)
                    if error == "missing_key":
                        answer = MISSING_KEY_MESSAGE
                        context_chunks = []
                    elif error:
                        # Log the real reason to the server console/logs and
                        # keep it in session_state so it can be inspected
                        # from the sidebar debug panel below.
                        print(f"[Groq generate_answer error] {error}")
                        st.session_state.last_groq_error = error
                        if error.startswith("rate_limit::"):
                            answer = get_rate_limit_message(language, error)
                        else:
                            answer = GROQ_ERROR_MESSAGE
                        context_chunks = []

    elapsed_time = round(time.time() - start_time, 2)
    audio_bytes = generate_tts(answer, language)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "audio": audio_bytes,
            "sources": context_chunks,
            "time": elapsed_time,
        }
    )


def handle_voice(audio_bytes):
    language = st.session_state.language
    with st.spinner("Transcribing audio question..."):
        transcript, error = transcribe_audio(audio_bytes, language)

    if error or not transcript:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error or "The voice recording could not be understood. Please try again.",
                "audio": None,
                "sources": [],
                "time": 0.0,
            }
        )
        return

    handle_question(transcript, label="🎙️ Voice Question:")


# ---------------------------------------------------------------------------
# Custom Header Render
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-top-row">
                <h1>Healthcare Assistant</h1>
                <div class="creator-badge">🟢 Created by Areeba Imran</div>
            </div>
            <p class="hero-welcome">Welcome to Healthcare Assistant.</p>
            <p class="hero-sub">Your trusted companion for hospital services, specialist availability, and department guidance,anytime.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status = st.session_state.kb_status
    if status == "no_pdfs":
        st.warning("No knowledge base documents found. Please add PDF files to the knowledge_base folder.")
    elif status in ("faiss_error", "embedding_error"):
        st.error("System index build error. Please click 'Rebuild Index' in the sidebar.")

    if not os.getenv("GROQ_API_KEY"):
        st.warning(MISSING_KEY_MESSAGE)


def _strip_prompt_emoji(text):
    for prefix in ("🚨 ", "🕒 ", "🏥 ", "📋 ", "💳 ", "🩺 ", "🧪 ", "📜 ", "💊 ", "🚙 "):
        text = text.replace(prefix, "")
    return text


def _quick_prompt_on_change():
    """
    Callback fired ONLY when the user actually changes the selectbox value
    (not on every rerun). It immediately resets the widget back to the
    placeholder inside the callback, which is the safe point in the
    Streamlit lifecycle to mutate a widget's own session_state key.

    This is what prevents a previously selected quick question from
    re-firing on every rerun (which used to repeat the same answer and
    drown out answers to newly typed questions).
    """
    selected_option = st.session_state.get("quick_prompt_select")
    if selected_option and selected_option != QUICK_PROMPT_OPTIONS[0]:
        clean_query = _strip_prompt_emoji(selected_option)
        handle_question(clean_query)
    # Reset immediately so this exact selection can't retrigger on the
    # next rerun, and so the same question can be picked again later.
    st.session_state.quick_prompt_select = QUICK_PROMPT_OPTIONS[0]


def render_quick_prompt_panel():
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-title-bar faq">
                <div class="panel-icon">❓</div>
                <div>
                    <p class="panel-title">Frequently Asked Questions</p>
                    <p class="panel-subtitle">Quick reference — jump straight to a common topic</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="faq-caption">Pick a topic below to get an instant answer:</p>',
            unsafe_allow_html=True,
        )
        st.selectbox(
            "Select a frequent question to ask immediately:",
            QUICK_PROMPT_OPTIONS,
            key="quick_prompt_select",
            on_change=_quick_prompt_on_change,
            label_visibility="collapsed",
        )


def render_chat_messages():
    for turn in st.session_state.chat_history:
        role = turn["role"]
        with st.chat_message(role):
            st.markdown(turn["content"])
            if role == "assistant":
                if turn.get("audio"):
                    st.audio(turn["audio"], format="audio/mp3")
                if turn.get("sources"):
                    render_sources(turn["sources"])

                if turn.get("time"):
                    st.markdown(
                        f"""
                        <div class="response-meta">
                            <span>⚡ Response Time: {turn['time']}s</span> •
                            <span>🔒 Grounded Verification</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


def render_chatbot_panel():
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-title-bar chat">
                <div class="panel-icon">🤖</div>
                <div>
                    <p class="panel-title">Healthcare Assistant Bot</p>
                    <p class="panel-subtitle">● Online · Grounded AI</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_chat_messages()

        st.divider()

        # Unified Text Input Form inside the bot panel
        with st.form("chatbot_card_form", clear_on_submit=True):
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                typed_question = st.text_input(
                    "Type query...",
                    label_visibility="collapsed",
                    placeholder="Ask doctor availability, visiting hours, or hospital rules...",
                )
            with col_btn:
                submitted = st.form_submit_button("Send 💬", use_container_width=True)

            if submitted and typed_question.strip():
                handle_question(typed_question)
                st.rerun()

        # Integrated Mic Recorder underneath text input inside the panel
        try:
            from streamlit_mic_recorder import mic_recorder
            st.caption("🎙️ Or click below to ask via voice:")
            audio = mic_recorder(
                start_prompt="🔴 Tap to Speak",
                stop_prompt="🟩 Stop & Process",
                just_once=True,
                use_container_width=True,
                format="wav",
                key="chatbot_mic",
            )
            if audio and audio.get("bytes"):
                audio_hash = hashlib.md5(audio["bytes"]).hexdigest()
                if st.session_state.last_audio_hash != audio_hash:
                    st.session_state.last_audio_hash = audio_hash
                    handle_voice(audio["bytes"])
                    st.rerun()
        except Exception:
            st.caption("Voice recording component unavailable.")


# ---------------------------------------------------------------------------
# Custom Sidebar Render
# ---------------------------------------------------------------------------
def render_info_panel():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>Healthcare System</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="sidebar-section-label teal">🌐 LANGUAGE SETTINGS</p>', unsafe_allow_html=True)
        selected_lang = st.selectbox(
            "Select Response Language",
            LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state.language),
            label_visibility="collapsed",
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()

        st.markdown('<p class="sidebar-section-label amber" style="margin-top:1.2rem;">🏥 HOSPITAL INFO</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-card">
                <ul>
                    <li>OPD &amp; Emergency: 24/7 Care</li>
                    <li>Specialties: Cardiology, Neurology, Pediatrics</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Index Rebuild Trigger Button
        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding FAISS index from PDFs..."):
                load_or_build_knowledge_base(force_rebuild=True)
            st.rerun()

        # Chat History Clear Trigger Button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # Debug panel: shows the real reason the last Groq call failed,
        # if any. Safe to leave in — it's empty/collapsed when there's
        # no error, and only ever shows technical text, not user data.
        if st.session_state.get("last_groq_error"):
            with st.expander("🛠️ Last technical error (debug)"):
                st.code(st.session_state.last_groq_error)


# ---------------------------------------------------------------------------
# Custom Footer Render
# ---------------------------------------------------------------------------
def render_footer():
    st.markdown(
        """
        <div class="custom-footer">
            <h3>Healthcare Assistant • Grounded System</h3>
            <p>Designed &amp; Developed by Areeba Imran</p>
            <p class="fine-print">© 2026 All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
def main():
    inject_css()
    init_session_state()

    # Automatically build or load FAISS index on first boot
    if st.session_state.kb_status == "not_loaded":
        load_or_build_knowledge_base(force_rebuild=False)

    # Render App Layout
    render_header()
    render_info_panel()

    # Main Grid Layout
    col_main, col_side = st.columns([2.8, 1.2])

    with col_main:
        render_chatbot_panel()

    with col_side:
        render_quick_prompt_panel()

    render_footer()


if __name__ == "__main__":
    main()
