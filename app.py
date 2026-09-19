"""
Areeba Imran Healthcare Assistant
---------------------------------
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
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Areeba Imran Healthcare Assistant",
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
    {"label": "Emergency Policy", "query": "What are the emergency protocols and contact rules?"},
    {"label": "Visiting Hours", "query": "What are the visiting hours for general and ICU wards?"},
    {"label": "Available Departments", "query": "Which specialized medical departments are available?"},
    {"label": "Admission Requirements", "query": "What documents and steps are required for patient admission?"},
]

NOT_FOUND_MESSAGE = (
    "I couldn't find enough relevant information in the healthcare "
    "knowledge base to answer that question accurately."
)

KB_NOT_READY_MESSAGE = (
    "The knowledge base is not ready yet. Please add PDF "
    "documents to the knowledge_base folder and use 'Rebuild Knowledge Base'."
)

MISSING_KEY_MESSAGE = (
    "Groq API key is not configured. Please set the GROQ_API_KEY "
    "environment variable before using the assistant."
)

GROQ_ERROR_MESSAGE = (
    "The assistant is temporarily unable to generate an answer. Please wait a "
    "moment and try again."
)

SYSTEM_PROMPT_TEMPLATE = """You are the Areeba Imran Healthcare Assistant, an educational information \
assistant for a healthcare facility.

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

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

# ---------------------------------------------------------------------------
# Custom Styling (Light Yellow / Soft Cream Theme)
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bg-color: #fefce8;
            --card-bg: #fffde7;
            --hero-bg: #fef08a;
            --text-main: #422006;
            --text-sub: #713f12;
            --border-color: #fde047;
            --btn-bg: #eab308;
            --btn-hover: #ca8a04;
        }

        html, body, .stApp {
            background-color: var(--bg-color) !important;
            color: var(--text-main) !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Hero Section */
        .hero-card {
            background-color: var(--hero-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(161, 98, 7, 0.08);
        }
        .hero-card h1 {
            color: var(--text-main) !important;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .hero-card p {
            color: var(--text-sub) !important;
            font-size: 1rem;
            line-height: 1.6;
            margin: 0;
        }
        .creator-tag {
            text-align: right;
            font-size: 0.85rem;
            font-weight: 600;
            color: #854d0e;
            margin-top: 0.8rem;
        }

        /* Sidebar Customization */
        section[data-testid="stSidebar"] {
            background-color: #fef9c3 !important;
            border-right: 1px solid var(--border-color);
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }

        /* Input Fields & Select Boxes */
        div[data-baseweb="select"], div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #facc15 !important;
        }

        /* Buttons Styling */
        .stButton > button {
            background-color: var(--btn-bg) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.2rem !important;
            box-shadow: 0 2px 6px rgba(161, 98, 7, 0.15) !important;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: var(--btn-hover) !important;
            transform: translateY(-1px);
        }

        /* Sidebar Buttons Override */
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #fde047 !important;
            color: var(--text-main) !important;
            border: 1px solid #eab308 !important;
            width: 100%;
        }

        /* Chat Output Styling */
        [data-testid="stChatMessage"] {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 14px !important;
            padding: 1.2rem !important;
            margin-bottom: 1rem !important;
        }

        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cached loaders
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
# PDF Pipeline
# ---------------------------------------------------------------------------
def load_pdf_files():
    return sorted(glob.glob(os.path.join(KB_DIR, "*.pdf")))

def extract_pdf_text(path):
    try:
        import fitz
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

# ---------------------------------------------------------------------------
# Retrieval & Generation
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Speech Audio Functions
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
        segments, _ = model.transcribe(tmp_path, language=hint, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            return None, "No speech detected."
        return text, None
    except Exception:
        return None, "Voice recording could not be transcribed."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

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
# State Initialization
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "chat_history": [],
        "language": "English",
        "kb_index": None,
        "kb_chunks": None,
        "kb_status": "not_loaded",
        "last_audio_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ---------------------------------------------------------------------------
# Core Question Processing
# ---------------------------------------------------------------------------
def handle_question(query, label=None):
    query = (query or "").strip()
    if not query:
        return

    display_text = query if not label else f"{label}\n\n{query}"
    st.session_state.chat_history.append({"role": "user", "content": display_text})
    language = st.session_state.language

    with st.spinner("Processing request..."):
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

    audio_bytes = generate_tts(answer, language)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "audio": audio_bytes,
            "sources": context_chunks,
        }
    )

def handle_voice(audio_bytes):
    language = st.session_state.language
    with st.spinner("Transcribing audio..."):
        transcript, error = transcribe_audio(audio_bytes, language)

    if error or not transcript:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": error or "Could not process audio.", "audio": None, "sources": []}
        )
        return

    handle_question(transcript, label="Voice Input:")

# ---------------------------------------------------------------------------
# UI Render Functions
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <h1>Areeba Imran Healthcare Assistant</h1>
            <p>An advanced healthcare information retrieval platform designed to provide structured and grounded responses based on verified hospital documentations.</p>
            <div class="creator-tag">Created by Areeba Imran</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_quick_prompts():
    st.markdown("**Quick Prompts**")
    cols = st.columns(len(QUICK_PROMPTS))
    for idx, item in enumerate(QUICK_PROMPTS):
        with cols[idx]:
            if st.button(item["label"], key=f"quick_{idx}", use_container_width=True):
                handle_question(item["query"])
                st.rerun()

def render_chat_history():
    for turn in st.session_state.chat_history:
        role = turn["role"]
        with st.chat_message(role):
            st.markdown(turn["content"])
            if role == "assistant":
                if turn.get("audio"):
                    # Auto-playing latest audio response
                    st.audio(turn["audio"], format="audio/mp3", autoplay=True)
                if turn.get("sources"):
                    with st.expander("Knowledge Sources"):
                        for c in turn["sources"]:
                            st.write(f"Source Document: {c['source']}")
                            st.caption(c["text"])

def render_voice_input():
    st.markdown("**Voice Input**")
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        st.caption("Voice input component is missing.")
        return

    audio = mic_recorder(
        start_prompt="Record Voice",
        stop_prompt="Stop & Send",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="voice_recorder",
    )

    if audio and audio.get("bytes"):
        audio_hash = hashlib.md5(audio["bytes"]).hexdigest()
        if st.session_state.last_audio_hash != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            handle_voice(audio["bytes"])
            st.rerun()

def render_sidebar():
    with st.sidebar:
        st.subheader("System Control")
        st.session_state.language = st.selectbox("Language Option", LANGUAGE_OPTIONS)
        st.markdown("---")
        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding..."):
                load_or_build_knowledge_base(force_rebuild=True)
            st.success("Updated successfully.")
            st.rerun()

# ---------------------------------------------------------------------------
# Main App Execution
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

    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.subheader("Workspace")
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_audio_hash = None
            st.rerun()

    with st.form("input_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            query_text = st.text_input("Question Input", label_visibility="collapsed", placeholder="Enter your query here...")
        with col_btn:
            submitted = st.form_submit_button("Submit", use_container_width=True)

    if submitted and query_text.strip():
        handle_question(query_text)

    render_voice_input()
    render_chat_history()

if __name__ == "__main__":
    main()
