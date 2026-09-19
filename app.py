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
    layout="wide",
    initial_sidebar_state="collapsed",
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
# Styling (CSS injected once; never rendered as visible text)
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        html, body, .stApp {
            background: linear-gradient(135deg, #f5f2fa 0%, #ece3f2 20%, #cfc0dd 42%, #7d6a95 65%, #362e49 85%, #1d1828 100%);
            background-attachment: fixed;
            background-size: cover;
            min-height: 100vh;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main,
        .block-container {
            background: transparent !important;
        }
        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #211c30 0%, #2c2540 55%, #3a2f4d 100%);
            border-right: 1px solid rgba(214, 196, 230, 0.15);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #f5f2fa;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stCaption {
            color: #ded7ea;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(214, 196, 230, 0.22);
        }
        h1, h2, h3, h4 {
            color: #241f33;
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            letter-spacing: 0.2px;
        }
        p, span, label, .stMarkdown, .stCaption {
            color: #34293f;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(120, 100, 150, 0.20);
            border-radius: 14px;
        }
        .stButton > button {
            background: linear-gradient(120deg, #5c4a78, #8a6f8f);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background: linear-gradient(120deg, #6d5a8c, #9c7f9f);
            color: #ffffff;
        }
        [data-testid="stChatInput"] {
            border-radius: 12px;
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            border: 1px solid rgba(120, 100, 150, 0.15);
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
    """Return a sorted list of PDF file paths found in the knowledge base folder."""
    return sorted(glob.glob(os.path.join(KB_DIR, "*.pdf")))


def extract_pdf_text(path):
    """Extract raw text from a PDF using PyMuPDF. Returns (text, error)."""
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
    """Normalize whitespace and strip unwanted control characters."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def chunk_text(text, source_name, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split cleaned text into overlapping word-based chunks tagged with their source file."""
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
    """Populate st.session_state with a ready FAISS index and chunk list."""
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
# Voice transcription (faster-whisper)
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
# Core question handling
# ---------------------------------------------------------------------------
def handle_question(query, label=None):
    query = (query or "").strip()
    if not query:
        return

    display_text = query if not label else f"{label}\n\n{query}"
    st.session_state.chat_history.append({"role": "user", "content": display_text})

    language = st.session_state.language

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
    with st.container(border=True):
        st.title("AreebaCare Healthcare Assistant")
        st.caption("Reliable healthcare information, grounded in the AreebaCare knowledge base.")

    status = st.session_state.kb_status
    if status == "no_pdfs":
        st.warning("No knowledge-base documents were found. Please add PDF files to the knowledge_base folder.")
    elif status in ("faiss_error", "embedding_error"):
        st.error("There was a problem preparing the knowledge base. Please try rebuilding it.")

    if not os.getenv("GROQ_API_KEY"):
        st.warning(MISSING_KEY_MESSAGE)

    if st.session_state.kb_failed:
        with st.expander("Some documents could not be processed"):
            for fname in st.session_state.kb_failed:
                st.write(f"- {fname}")


def render_controls():
    col_lang, col_rebuild, col_clear = st.columns([2, 1, 1])
    with col_lang:
        st.session_state.language = st.selectbox(
            "Language", LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index(st.session_state.language)
        )
    with col_rebuild:
        st.write("")
        if st.button("Rebuild Knowledge Base", use_container_width=True):
            load_or_build_knowledge_base(force_rebuild=True)
            st.rerun()
    with col_clear:
        st.write("")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_audio_hash = None
            st.rerun()


def render_example_questions():
    with st.expander("Try asking"):
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
                    st.markdown("**Audio Response**")
                    st.audio(turn["audio"], format="audio/mp3")
                if turn.get("sources"):
                    render_sources(turn["sources"])


def render_voice_input():
    st.markdown("**Voice Question**")
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        st.caption("Voice input is currently unavailable in this environment.")
        return

    audio = mic_recorder(
        start_prompt="Record",
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
        st.subheader("Information Panel")

        with st.container(border=True):
            st.subheader("AreebaCare")
            st.write(
                "A fictional educational healthcare information assistant designed "
                "to answer questions using a controlled hospital knowledge base."
            )

        with st.container(border=True):
            st.subheader("Response Flow")
            st.write("Question")
            st.write("\u2192 Retrieval")
            st.write("\u2192 Grounded Answer")
            st.write("\u2192 Audio Response")

        with st.container(border=True):
            st.subheader("Knowledge Base")
            if st.session_state.kb_status == "ready":
                st.write(f"{st.session_state.kb_loaded_count} documents")
                st.caption("Knowledge base ready")
            else:
                st.caption("Knowledge base not ready yet")
            st.write("PDF-based")
            st.write("FAISS retrieval")
            st.write("Grounded responses")

        with st.container(border=True):
            st.subheader("Supported Languages")
            for lang in LANGUAGE_OPTIONS:
                st.write(lang)

        with st.container(border=True):
            st.subheader("Safety")
            st.write("Educational information only.")
            st.write("No diagnosis.")
            st.write("No treatment prescription.")
            st.write("No invented medical information.")


def render_footer():
    st.divider()
    st.caption("AreebaCare Healthcare Assistant \u2022 Educational Project")
    st.caption("Created by Areeba Imran")
    st.caption("Designed & Developed by Areeba Imran")
    st.caption("\u00a9 2026 Areeba Imran. All rights reserved.")


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
    render_controls()

    st.subheader("Main Assistant")
    render_example_questions()

    with st.container(border=True):
        st.write("Ask AreebaCare a question")
        with st.form("text_question_form", clear_on_submit=True):
            col_input, col_submit = st.columns([5, 1])
            with col_input:
                typed_question = st.text_input(
                    "Ask AreebaCare a question...",
                    label_visibility="collapsed",
                    placeholder="Ask AreebaCare a question...",
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
