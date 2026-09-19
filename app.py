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
GROQ_FALLBACK_MODEL_NAME = "llama-3.1-8b-instant"

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
# Custom Vibrant Slate & Light Red Chatbot UI
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --primary-bg: #f3f7f6;
            --hero-gradient: linear-gradient(135deg, #112a2e 0%, #1a3d42 50%, #224d53 100%);
            --btn-gradient: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
            --btn-hover: linear-gradient(135deg, #115e59 0%, #0f766e 100%);
            --user-msg-bg: #ffe4e6;
            --assistant-msg-bg: #ffffff;
            --border-color: #fecdd3;
            --text-dark: #0f172a;
        }

        html, body, .stApp {
            background-color: var(--primary-bg) !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-dark);
        }

        /* Hero Banner */
        .hero-banner {
            background: var(--hero-gradient);
            border-radius: 20px;
            padding: 1.8rem 2.2rem;
            margin-bottom: 1.5rem;
            color: #ffffff;
            box-shadow: 0 14px 35px -10px rgba(17, 42, 46, 0.45);
            border: 1px solid rgba(153, 246, 228, 0.2);
        }
        .hero-top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.6rem;
        }
        .hero-banner h1 {
            color: #ffffff !important;
            margin: 0;
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .hero-banner p {
            color: #ccfbf1 !important;
            margin: 0;
            font-size: 0.98rem;
            line-height: 1.5;
            max-width: 820px;
        }

        /* Creator Badge */
        .creator-badge {
            background: rgba(45, 212, 191, 0.12);
            border: 1px solid rgba(45, 212, 191, 0.4);
            color: #2dd4bf !important;
            padding: 0.4rem 0.9rem;
            border-radius: 30px;
            font-size: 0.82rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            backdrop-filter: blur(8px);
            white-space: nowrap;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #112a2e !important;
            border-right: 1px solid rgba(153, 246, 228, 0.15);
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {
            color: #f0fdfa !important;
        }

        .sidebar-brand {
            padding: 0.5rem 0 1rem 0;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(153, 246, 228, 0.2);
        }
        .sidebar-brand h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
            color: #ffffff !important;
        }

        .sidebar-info-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(153, 246, 228, 0.25);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.9rem;
        }
        .sidebar-info-card h4 {
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
            color: #2dd4bf !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .sidebar-info-card ul {
            margin: 0;
            padding-left: 1.1rem;
            font-size: 0.85rem;
            line-height: 1.4;
            color: #f0fdfa !important;
        }
        .sidebar-info-card li {
            color: #f0fdfa !important;
        }

        /* Standard Buttons */
        .stButton > button {
            background: var(--btn-gradient) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.2rem !important;
            box-shadow: 0 4px 14px rgba(13, 148, 136, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button:hover {
            background: var(--btn-hover) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(13, 148, 136, 0.4) !important;
        }

        /* Sidebar Buttons */
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(45, 212, 191, 0.15) !important;
            border: 1px solid rgba(45, 212, 191, 0.4) !important;
            color: #ffffff !important;
            width: 100%;
            box-shadow: none !important;
        }

        /* Selectbox Styling */
        div[data-baseweb="select"] {
            border-radius: 10px !important;
            border: 1.5px solid #0d9488 !important;
            background-color: #ffffff !important;
        }
        div[data-baseweb="select"] * {
            color: #0f172a !important;
        }

        /* --- LIGHT RED DISTINCT CHATBOT CONTAINER --- */
        .chatbot-container {
            background: #fff1f2;
            border: 2px solid #fecdd3;
            border-radius: 20px;
            box-shadow: 0 12px 30px rgba(225, 29, 72, 0.08);
            padding: 1.5rem;
            margin: 0 auto 2rem auto;
            max-width: 900px;
        }

        .chatbot-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, #881337 0%, #9f1239 50%, #be123c 100%);
            padding: 0.9rem 1.3rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 18px rgba(159, 18, 57, 0.25);
        }

        .chatbot-avatar {
            width: 42px;
            height: 42px;
            background: #ffe4e6;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border: 2px solid #fecdd3;
        }

        /* Chat Messages inside Red Container */
        [data-testid="stChatMessage"] {
            border-radius: 14px !important;
            padding: 1rem !important;
            margin-bottom: 0.9rem !important;
        }
        [data-testid="stChatMessage"]:nth-child(even) {
            background-color: #ffe4e6 !important;
            border: 1px solid #fecdd3 !important;
        }
        [data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #ffffff !important;
            border: 1px solid #ffe4e6 !important;
            box-shadow: 0 2px 10px rgba(159, 18, 57, 0.04) !important;
        }

        .response-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.78rem;
            color: #be123c;
            font-weight: 600;
            margin-top: 0.6rem;
            padding-top: 0.4rem;
            border-top: 1px dashed #fecdd3;
        }

        /* Footer Styling */
        .custom-footer {
            background: var(--hero-gradient);
            border-radius: 14px;
            padding: 1.1rem;
            text-align: center;
            color: #ccfbf1;
            margin-top: 2rem;
            font-size: 0.85rem;
            box-shadow: 0 8px 20px rgba(17, 42, 46, 0.2);
            border: 1px solid rgba(153, 246, 228, 0.2);
        }
        .custom-footer p {
            margin: 0.2rem 0;
        }

        footer {
            visibility: hidden;
        }
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
        if type(e).__name__ == "RateLimitError" or status_code == 429:
            return None, f"rate_limit::{e}"
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


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

    if error and error.startswith("rate_limit::") and GROQ_FALLBACK_MODEL_NAME:
        # Primary model's daily quota is exhausted — the fallback model has
        # its own separate quota on Groq's free tier, so try it before
        # giving up.
        print(f"[Groq] {GROQ_MODEL_NAME} rate-limited, falling back to {GROQ_FALLBACK_MODEL_NAME}")
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
            <p>Welcome to Healthcare Assistant. Access verified hospital policies, specialist availability, and clinical department guidelines in real time.</p>
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

    This is what fixes the bug where the previously selected quick
    question kept re-firing on every rerun (causing the same answer to
    repeat and drowning out answers to newly typed questions).
    """
    selected_option = st.session_state.get("quick_prompt_select")
    if selected_option and selected_option != QUICK_PROMPT_OPTIONS[0]:
        clean_query = _strip_prompt_emoji(selected_option)
        handle_question(clean_query)
    # Reset immediately so this exact selection can't retrigger on the
    # next rerun, and so the same question can be picked again later.
    st.session_state.quick_prompt_select = QUICK_PROMPT_OPTIONS[0]


def render_quick_prompt_dropdown():
    st.markdown("**🏥 Frequently Asked Questions**")
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


def render_chatbot_card():
    # Light Red / Rose Styled Chatbot Card Container
    st.markdown(
        """
        <div class="chatbot-header">
            <div style="display:flex; align-items:center; gap:12px;">
                <div class="chatbot-avatar">🤖</div>
                <div>
                    <h3 style="margin:0; font-size:1.15rem; color:#ffffff;">Healthcare Assistant Bot</h3>
                    <span style="font-size:0.75rem; color:#fecdd3;">● Online | Grounded AI</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chat history viewport inside card
    render_chat_messages()

    st.markdown("<hr style='margin:1rem 0; border-color:#fecdd3;'>", unsafe_allow_html=True)

    # Unified Text Input Form inside the bot card
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

    # Integrated Mic Recorder underneath text input inside card
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

        st.markdown("### 🌐 Language Settings")
        selected_lang = st.selectbox(
            "Select Response Language",
            LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state.language),
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()

        st.markdown("---")

        # Hospital Info Card
        st.markdown(
            """
            <div class="sidebar-info-card">
                <h4>🏥 Hospital Info</h4>
                <ul>
                    <li>OPD & Emergency: 24/7 Care</li>
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
            <h3 style="margin: 0; font-size: 1.1rem; color: #ffffff;">Healthcare Assistant • Grounded System</h3>
            <p style="margin: 0.3rem 0; font-weight: 500;">Designed & Developed by Areeba Imran</p>
            <p style="font-size: 0.78rem; opacity: 0.85; margin-top: 0.4rem;">© 2026 All rights reserved.</p>
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
        render_chatbot_card()

    with col_side:
        render_quick_prompt_dropdown()

    render_footer()


if __name__ == "__main__":
    main()
