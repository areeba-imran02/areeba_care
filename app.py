"""
AreebaCare Healthcare Assistant
--------------------------------
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

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AreebaCare Healthcare Assistant",
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

EXAMPLE_QUESTIONS = [
    "What are the registration requirements?",
    "What are the visiting hours?",
    "What departments are available?",
    "What documents are needed for admission?",
    "What is the billing and insurance policy?",
    "How can I access my medical records?",
    "What are the patient rights?",
]

NOT_FOUND_MESSAGE = (
    "I couldn't find enough relevant information in the AreebaCare "
    "knowledge base to answer that question accurately."
)

KB_NOT_READY_MESSAGE = (
    "The AreebaCare knowledge base is not ready yet. Please add PDF "
    "documents to the knowledge_base folder and use \"Rebuild Knowledge "
    "Base\" to prepare the assistant."
)

MISSING_KEY_MESSAGE = (
    "Groq API key is not configured. Please set the GROQ_API_KEY "
    "environment variable before using the assistant."
)

GROQ_ERROR_MESSAGE = (
    "AreebaCare is temporarily unable to generate an answer. Please wait a "
    "moment and try again."
)

SYSTEM_PROMPT_TEMPLATE = """You are the AreebaCare Healthcare Assistant, an educational information \
assistant for a fictional hospital called AreebaCare.

STRICT RULES YOU MUST ALWAYS FOLLOW:
1. Answer ONLY using the AreebaCare knowledge base context provided below. Never use outside or general medical knowledge.
2. Never invent, guess, or assume information that is not explicitly present in the provided context, including prices, timings, departments, doctors, medical services, policies, procedures, contact information, insurance rules, medicine information, or hospital facilities.
3. If the provided context does not contain enough information to answer the question, clearly say so instead of guessing.
4. Never diagnose a medical condition.
5. Never prescribe medication or dosages.
6. Never recommend a specific treatment.
7. Never claim to replace a licensed doctor or medical professional.
8. If the question describes a potential medical emergency, advise the user to seek immediate professional or emergency medical care, without inventing hospital-specific emergency instructions that are not present in the context.
9. Keep answers clear, concise, and genuinely useful.
10. {language_instruction}
11. Remember that AreebaCare is a fictional, educational hospital knowledge base used for demonstration purposes only.

CONTEXT FROM AREEBACARE KNOWLEDGE BASE:
{context}
"""

# ---------------------------------------------------------------------------
# Custom Vibrant & Refined Styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --primary-dark: #0f172a;
            --accent-blue: #1d4ed8;
            --accent-hover: #1e40af;
            --bg-color: #f0f7ff; /* Soft ice-blue backdrop */
            --card-bg: #ffffff;
            --text-dark: #0f172a;
            --border-color: #bfdbfe;
        }

        html, body, .stApp {
            background-color: var(--bg-color) !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Hero Banner */
        .areebacare-hero {
            background: linear-gradient(135deg, #0284c7 0%, #1e40af 60%, #0f172a 100%);
            border-radius: 18px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 1.8rem;
            color: #ffffff;
            box-shadow: 0 10px 25px -5px rgba(30, 64, 175, 0.25);
        }
        .areebacare-hero h1 {
            color: #ffffff !important;
            margin: 0 0 0.3rem 0;
            font-size: 2.3rem;
            font-weight: 700;
        }
        .areebacare-hero .creator-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            color: #e0f2fe;
            padding: 0.25rem 0.85rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        .areebacare-hero p {
            color: #e0f2fe !important;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.6;
            max-width: 820px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0b1329 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        .sidebar-brand {
            padding: 1rem 0;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        }
        .sidebar-brand h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
            color: #ffffff !important;
        }
        .sidebar-brand p {
            font-size: 0.85rem;
            color: #38bdf8 !important;
            margin: 0.2rem 0 0 0;
            font-weight: 500;
        }

        .sidebar-info-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.2rem;
        }
        .sidebar-info-card h4 {
            font-size: 0.92rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
            color: #38bdf8 !important;
            text-transform: uppercase;
        }
        .sidebar-info-card p, .sidebar-info-card ul {
            font-size: 0.88rem;
            color: #cbd5e1 !important;
            margin: 0;
            line-height: 1.5;
        }
        .sidebar-info-card ul {
            padding-left: 1.2rem;
        }

        /* Styled Language Select Box */
        div[data-baseweb="select"] {
            border-radius: 12px !important;
            border: 2px solid #93c5fd !important;
            background-color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(29, 78, 216, 0.08) !important;
        }
        div[data-baseweb="select"] * {
            color: #0f172a !important;
            font-weight: 600 !important;
        }

        /* Main Buttons Custom Style */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.3rem !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
        }

        /* Sidebar Buttons */
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
            width: 100%;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.18) !important;
        }

        /* Chat Cards Styling */
        [data-testid="stChatMessage"] {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04) !important;
            padding: 1.25rem !important;
            margin-bottom: 1rem !important;
        }

        /* Input Controls */
        [data-testid="stTextInput"] input {
            border-radius: 10px !important;
            border: 2px solid #bfdbfe !important;
            padding: 0.7rem 1rem !important;
            background-color: #ffffff !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
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
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception:
        return None


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
def generate_answer(query, context_chunks, language, history):
    api_key = os.getenv("GROQ_API_KEY")
    client = get_groq_client(api_key)
    if client is None:
        return None, "missing_key"

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
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
        answer = response.choices[0].message.content.strip()
        return answer, None
    except Exception as e:
        return None, str(e)


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
    with st.expander("Knowledge Sources"):
        for c in context_chunks:
            st.markdown(f"**{c['source']}**")
            st.caption(f"Relevance: {c['score']:.2f}")
            with st.expander(f"View retrieved text from {c['source']}", expanded=False):
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
        "pending_example": None,
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
        return "السلام علیکم! میں AreebaCare کا ہیلتھ کیئر اسسٹنٹ ہوں۔ میں آپ کی کیا مدد کر سکتا ہوں؟ آپ ہسپتال کی سہولیات، رجسٹریشن اور دیگر معلومات کے بارے میں پوچھ سکتے ہیں۔"
    elif language == "Roman Urdu":
        return "Aoa! Main AreebaCare Healthcare Assistant hoon. Main aap ki kya madad kar sakta hoon? Aap hospital ki visiting hours, registration ya kisi bhi policy ke baray mein pooch saktay hain."
    else:
        return "Hello! Welcome to AreebaCare. How can I assist you today? You can ask me about hospital services, visiting hours, registration, and patient guidelines."


# ---------------------------------------------------------------------------
# Core question handling
# ---------------------------------------------------------------------------
def handle_question(query, label=None):
    query = (query or "").strip()
    if not query:
        return

    display_text = query if not label else f"{label}\n\n{query}"
    st.session_state.chat_history.append({"role": "user", "content": display_text})

    language = st.session_state.language

    # Quick Greeting Handling
    if is_greeting(query):
        answer = get_greeting_response(language)
        context_chunks = []
    else:
        with st.spinner("Retrieving knowledge and generating answer..."):
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
    with st.spinner("Transcribing your voice question..."):
        transcript, error = transcribe_audio(audio_bytes, language)

    if error or not transcript:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error or "The voice recording could not be understood. Please try again.",
                "audio": None,
                "sources": [],
            }
        )
        return

    handle_question(transcript, label="Your voice question:")


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="areebacare-hero">
            <div class="creator-badge">Created by Areeba Imran</div>
            <h1>AreebaCare Healthcare Assistant</h1>
            <p>Welcome to <b>AreebaCare</b> — your intelligent, retrieval-augmented healthcare information system. 
            Designed to deliver fast, grounded, and accurate answers directly from official hospital knowledge bases, 
            empowering patients with instant insights on registration, policies, departments, and general guidelines.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status = st.session_state.kb_status
    if status == "no_pdfs":
        st.warning("No knowledge base documents found. Please insert PDF files into the knowledge_base folder.")
    elif status in ("faiss_error", "embedding_error"):
        st.error("System index build error. Please click 'Rebuild Index' in the sidebar.")

    if not os.getenv("GROQ_API_KEY"):
        st.warning(MISSING_KEY_MESSAGE)


def render_language_selector():
    col_lang, _ = st.columns([1.2, 2.8])
    with col_lang:
        st.session_state.language = st.selectbox(
            "🌐 Choose Response Language", 
            LANGUAGE_OPTIONS, 
            index=LANGUAGE_OPTIONS.index(st.session_state.language)
        )


def render_example_questions():
    with st.expander("Suggested Questions"):
        for i, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                handle_question(question)
                st.rerun()


def render_chat_history():
    for turn in st.session_state.chat_history:
        role = turn["role"]
        with st.chat_message(role):
            st.markdown(turn["content"])
            if role == "assistant":
                if turn.get("audio"):
                    st.audio(turn["audio"], format="audio/mp3")
                if turn.get("sources"):
                    render_sources(turn["sources"])


def render_voice_input():
    st.markdown("**Voice Input**")
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        st.caption("Voice recording is unavailable in this environment.")
        return

    audio = mic_recorder(
        start_prompt="Record Question",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="areebacare_mic",
    )

    if audio and audio.get("bytes"):
        audio_hash = hashlib.md5(audio["bytes"]).hexdigest()
        if st.session_state.last_audio_hash != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            handle_voice(audio["bytes"])
            st.rerun()


def render_info_panel():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>Areeba Care</h2>
                <p>Created by Areeba Imran</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="sidebar-info-card">
                <h4>System Architecture</h4>
                <p>Built with Retrieval-Augmented Generation (RAG) to deliver verified answers directly from structured medical documentation.</p>
            </div>

            <div class="sidebar-info-card">
                <h4>Safety Protocols</h4>
                <ul>
                    <li>Grounded strictly on provided data</li>
                    <li>No medical diagnosis</li>
                    <li>No prescription generation</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        
        if st.button("Rebuild Search Index", use_container_width=True):
            with st.spinner("Rebuilding knowledge base..."):
                load_or_build_knowledge_base(force_rebuild=True)
            st.success("Index updated successfully.")
            st.rerun()


def render_footer():
    st.divider()
    st.caption("AreebaCare Healthcare Assistant • Educational Demonstration")
    st.caption("Created by Areeba Imran")
    st.caption("© 2026 Areeba Imran. All rights reserved.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    inject_css()
    init_session_state()

    if st.session_state.kb_status == "not_loaded":
        load_or_build_knowledge_base(force_rebuild=False)

    render_info_panel()

    render_header()
    render_language_selector()

    col_heading, col_clear = st.columns([4, 1])
    with col_heading:
        st.subheader("Interactive Assistant")
    with col_clear:
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_audio_hash = None
            st.rerun()

    render_example_questions()

    with st.container():
        with st.form("text_question_form", clear_on_submit=True):
            col_input, col_submit = st.columns([5, 1])
            with col_input:
                typed_question = st.text_input(
                    "Ask AreebaCare a question...",
                    label_visibility="collapsed",
                    placeholder="Ask about visiting hours, registration, procedures...",
                )
            with col_submit:
                submitted = st.form_submit_button("Ask", use_container_width=True)
        if submitted and typed_question.strip():
            handle_question(typed_question)

    render_voice_input()
    render_chat_history()

    render_footer()


if __name__ == "__main__":
    main()
