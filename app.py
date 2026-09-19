"""
Healthcare Assistant
--------------------
A grounded, retrieval-augmented healthcare information assistant built on top
of a local PDF knowledge base.
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
# Page configuration
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

CHUNK_SIZE_WORDS = 600
CHUNK_OVERLAP_WORDS = 100
TOP_K = 4
RELEVANCE_THRESHOLD = 0.30

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

QUICK_PROMPTS = [
    {"label": "🚨 Emergency Policy", "query": "What are the emergency protocols and contact rules?"},
    {"label": "🕒 Visiting Hours", "query": "What are the visiting hours for general and ICU wards?"},
    {"label": "🏥 Medical Departments", "query": "Which specialized medical departments are available?"},
    {"label": "📋 Patient Admission", "query": "What documents and steps are required for patient admission?"},
    {"label": "💳 Insurance & Billing", "query": "What insurance and billing procedures are supported?"},
]

NOT_FOUND_MESSAGE = (
    "I couldn't find enough relevant information in the hospital "
    "knowledge base to answer that question accurately."
)

KB_NOT_READY_MESSAGE = (
    "The knowledge base is not ready yet. Please add PDF "
    "documents to the knowledge_base folder and use \"Rebuild Knowledge Base\"."
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
# Custom Vibrant Theme
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        html, body, .stApp {
            background-color: #f8fafc !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #0f172a;
        }

        /* Hero Banner Box */
        .hero-banner {
            background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #0f766e 100%);
            border-radius: 16px;
            padding: 2.2rem;
            margin-bottom: 1.5rem;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(4, 120, 87, 0.25);
            position: relative;
        }
        .hero-banner h1 {
            color: #ffffff !important;
            margin: 0 0 0.5rem 0;
            font-size: 2.3rem;
            font-weight: 700;
        }
        .hero-banner p {
            color: #e6f4f1 !important;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.6;
            max-width: 850px;
        }
        .hero-banner .creator-badge {
            margin-top: 1.2rem;
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #ffffff !important;
            padding: 0.35rem 0.9rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Sidebar Customization */
        section[data-testid="stSidebar"] {
            background-color: #1e293b !important;
            border-right: 1px solid #334155;
        }
        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        /* Fix visibility for selectbox inside sidebar */
        section[data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #0f172a !important;
        }

        .sidebar-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .sidebar-card h4 {
            color: #38bdf8 !important;
            font-size: 0.92rem;
            margin: 0 0 0.6rem 0;
            text-transform: uppercase;
        }
        .sidebar-card ul {
            margin: 0;
            padding-left: 1.1rem;
            font-size: 0.85rem;
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #0d9488 0%, #059669 100%) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #0f766e 0%, #047857 100%) !important;
            transform: translateY(-1px);
        }

        /* Sidebar Button */
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.12) !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            color: #ffffff !important;
        }

        /* Chat Output Messages */
        [data-testid="stChatMessage"] {
            border-radius: 14px !important;
            padding: 1.2rem !important;
            margin-bottom: 1rem !important;
        }
        [data-testid="stChatMessage"]:nth-child(even) {
            background-color: #f0fdf4 !important;
            border: 1px solid #bbf7d0 !important;
        }
        [data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
        }

        /* Footer Banner */
        .custom-footer {
            background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            color: #ffffff;
            margin-top: 2rem;
            font-size: 0.9rem;
        }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Model Loaders
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
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF Processing & Retrieval
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
        chunk_str = " ".join(words[start:end]).strip()
        if chunk_str:
            chunks.append({"text": chunk_str, "source": source_name})
        if end == len(words):
            break
        start += step
    return chunks


def build_embeddings(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
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
            st.session_state.kb_status = "ready"
            return

    pdf_paths = load_pdf_files()
    if not pdf_paths:
        st.session_state.kb_index = None
        st.session_state.kb_chunks = None
        st.session_state.kb_status = "no_pdfs"
        return

    try:
        model = get_embedding_model()
    except Exception:
        st.session_state.kb_status = "embedding_error"
        return

    all_chunks = []
    for path in pdf_paths:
        fname = os.path.basename(path)
        text, err = extract_pdf_text(path)
        if err or not text:
            continue
        cleaned = clean_text(text)
        if cleaned:
            all_chunks.extend(chunk_text(cleaned, fname))

    if not all_chunks:
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
    st.session_state.kb_status = "ready"


def retrieve_context(query, index, chunks, model, top_k=TOP_K, threshold=RELEVANCE_THRESHOLD):
    if index is None or not chunks:
        return []
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1 or score < threshold:
            continue
        c = chunks[idx]
        results.append({"text": c["text"], "source": c["source"], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# Answer Generation & Voice
# ---------------------------------------------------------------------------
def generate_answer(query, context_chunks, language, history):
    api_key = os.getenv("GROQ_API_KEY")
    client = get_groq_client(api_key)
    if client is None:
        return None, "missing_key"

    context_text = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        language_instruction=LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["English"]),
        context=context_text,
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=900,
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)


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


def transcribe_audio(audio_bytes, language):
    model = get_whisper_model()
    if model is None:
        return None, "Voice transcription model is not available."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        hint = WHISPER_LANG_HINT.get(language)
        segments, _ = model.transcribe(tmp_path, language=hint, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return (text, None) if text else (None, "No voice detected.")
    except Exception:
        return None, "Could not process audio."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass


# ---------------------------------------------------------------------------
# Core Question Handler
# ---------------------------------------------------------------------------
def is_greeting(query):
    cleaned = re.sub(r"[^\w\s]", "", query.lower().strip())
    return cleaned in {"hi", "hello", "hey", "salam", "aoa", "assalam o alaikum"}


def handle_question(query):
    query = (query or "").strip()
    if not query:
        return

    start_time = time.time()
    st.session_state.chat_history.append({"role": "user", "content": query})
    language = st.session_state.language

    if is_greeting(query):
        if language == "Urdu":
            answer = "السلام علیکم! میں آپ کا Healthcare Assistant ہوں۔ آپ مجھ سے ہسپتال کی خدمات اور پالیسیوں کے بارے میں پوچھ سکتے ہیں۔"
        elif language == "Roman Urdu":
            answer = "Aoa! Main aap ka Healthcare Assistant hoon. Aap hospital policies aur services ke baare mein pooch sakte hain."
        else:
            answer = "Hello! Welcome to Healthcare Assistant. How can I help you today with hospital policies and services?"
        context_chunks = []
    else:
        with st.spinner("Searching knowledge base..."):
            if st.session_state.kb_status != "ready" or st.session_state.kb_index is None:
                answer = KB_NOT_READY_MESSAGE
                context_chunks = []
            else:
                model = get_embedding_model()
                context_chunks = retrieve_context(query, st.session_state.kb_index, st.session_state.kb_chunks, model)
                if not context_chunks:
                    answer = NOT_FOUND_MESSAGE
                else:
                    answer, error = generate_answer(query, context_chunks, language, st.session_state.chat_history)
                    if error == "missing_key":
                        answer = MISSING_KEY_MESSAGE
                        context_chunks = []
                    elif error:
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


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def init_session_state():
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "language" not in st.session_state: st.session_state.language = "English"
    if "kb_index" not in st.session_state: st.session_state.kb_index = None
    if "kb_chunks" not in st.session_state: st.session_state.kb_chunks = None
    if "kb_status" not in st.session_state: st.session_state.kb_status = "not_loaded"
    if "last_audio_hash" not in st.session_state: st.session_state.last_audio_hash = None


def render_header():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Healthcare Assistant</h1>
            <p>Welcome to <b>Healthcare Assistant</b> — an advanced, retrieval-augmented healthcare information system. 
            Designed to deliver fast, verified, and grounded answers directly from hospital documentation, 
            empowering users with accurate guidance on registration, services, policies, and care.</p>
            <div class="creator-badge">Created by Areeba Imran</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.title("🏥 Healthcare AI")
        
        st.markdown("### 🌐 Select Language")
        st.session_state.language = st.selectbox(
            "Language Options",
            LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state.language),
            label_visibility="collapsed"
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <h4>⚡ AI Capabilities & Models</h4>
                <ul>
                    <li><b>Vector Index:</b> FAISS + Sentence-Transformers</li>
                    <li><b>Language Model:</b> Groq (GPT-OSS-120B)</li>
                    <li><b>Voice Processing:</b> OpenAI Whisper ASR</li>
                    <li><b>Text-To-Speech:</b> gTTS Synthesis</li>
                </ul>
            </div>
            
            <div class="sidebar-card">
                <h4>🔒 Grounded Safety</h4>
                <ul>
                    <li>Strictly factual KB responses</li>
                    <li>No hallucinated medical claims</li>
                    <li>No medical diagnosis/prescriptions</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            with st.spinner("Rebuilding Index..."):
                load_or_build_knowledge_base(force_rebuild=True)
            st.rerun()


def render_quick_prompts():
    st.markdown("**💡 Quick Sample Questions**")
    cols = st.columns(len(QUICK_PROMPTS))
    for idx, item in enumerate(QUICK_PROMPTS):
        with cols[idx]:
            if st.button(item["label"], key=f"qp_{idx}", use_container_width=True):
                handle_question(item["query"])
                st.rerun()


def render_chat_history():
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                if turn.get("audio"):
                    st.audio(turn["audio"], format="audio/mp3")
                if turn.get("sources"):
                    with st.expander("📚 Verified Sources"):
                        for c in turn["sources"]:
                            st.caption(f"Source: {c['source']} (Score: {c['score']:.2f})")
                            st.write(c["text"])


# ---------------------------------------------------------------------------
# Main App Structure
# ---------------------------------------------------------------------------
def main():
    inject_css()
    init_session_state()

    if st.session_state.kb_status == "not_loaded":
        load_or_build_knowledge_base(force_rebuild=False)

    render_sidebar()
    render_header()

    render_quick_prompts()
    st.markdown("---")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Assistant Chat Workspace")
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # User Input Field
    user_input = st.chat_input("Ask Healthcare Assistant a question...")
    if user_input:
        handle_question(user_input)
        st.rerun()

    render_chat_history()

    st.markdown(
        """
        <div class="custom-footer">
            Healthcare Assistant • Grounded Information System<br>
            Designed & Developed by <b>Areeba Imran</b> | © 2026 All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
