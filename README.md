# AreebaCare Healthcare Assistant

**Reliable Healthcare Information, Grounded in AreebaCare Knowledge**

An educational, retrieval-augmented healthcare information assistant built with Streamlit. AreebaCare answers questions strictly from a local PDF knowledge base using a local RAG pipeline (PyMuPDF + Sentence-Transformers + FAISS), generates grounded answers with Groq, and produces spoken audio answers with gTTS. Voice questions are supported through `streamlit-mic-recorder` and `faster-whisper`.

> This is an educational healthcare information assistant, **not** a real medical diagnosis or treatment system. It does not diagnose conditions, prescribe medication, or replace a licensed doctor.

---

## 1. Project Overview

AreebaCare is a fictional hospital used purely for demonstration purposes. The assistant reads a set of AreebaCare policy PDFs, builds a local vector index, and answers user questions only using information actually found in those PDFs. If the knowledge base does not contain relevant information, the assistant says so instead of guessing.

## 2. Features

- Fully automatic local RAG pipeline — no manual FAISS setup required
- Text questions and voice questions, both producing text **and** audio answers automatically
- Multilingual support: English, Urdu, Roman Urdu
- Strict grounding — never invents prices, timings, departments, doctors, policies, or medical advice
- "Knowledge Sources" panel showing which PDF and similarity score backed each answer
- "Rebuild Knowledge Base" admin action for refreshing the index after PDFs change
- Clean chat-style conversation history with a "Clear Conversation" action
- Modern, professional two-column dashboard layout with no visible HTML/CSS/code
- Graceful error handling for missing API keys, missing PDFs, failed transcription, and failed TTS

## 3. Architecture

```text
PDF Knowledge Base
        |
PyMuPDF Text Extraction
        |
Text Cleaning
        |
Chunking (≈600 words, 100-word overlap)
        |
Sentence-Transformer Embeddings (all-MiniLM-L6-v2)
        |
FAISS Vector Index (cosine similarity via normalized inner product)
        |
User Question (typed or transcribed from voice)
        |
Question Embedding
        |
Similarity Search + Relevance Threshold
        |
Relevant Knowledge Chunks
        |
Strict Grounded Prompt
        |
Groq (openai/gpt-oss-120b)
        |
Text Answer
        |
gTTS
        |
Audio Answer
```

Voice input pipeline:

```text
Record Voice -> faster-whisper Transcription -> Show Transcript -> RAG Retrieval -> Groq Answer -> gTTS Audio
```

## 4. Folder Structure

```text
AreebaCare/
│
├── app.py                 # Complete Streamlit application (single file)
├── requirements.txt       # Python dependencies
├── README.md               # This file
│
├── knowledge_base/         # Place your AreebaCare PDF documents here
│   ├── Admission_and_Discharge_Policy.pdf
│   ├── Appointment_and_Registration_Policy.pdf
│   ├── Billing_and_Insurance_Policy.pdf
│   ├── Emergency_and_Triage_Policy.pdf
│   ├── Hospital_Departments_and_Services.pdf
│   ├── Hospital_Overview_and_Mission.pdf
│   ├── Infection_Control_Policy.pdf
│   ├── Laboratory_and_Diagnostic_Services.pdf
│   ├── Medical_Records_and_Privacy_Policy.pdf
│   ├── Patient_Rights_and_Responsibilities.pdf
│   ├── Pharmacy_and_Medication_Policy.pdf
│   └── Visitor_and_Attendant_Policy.pdf
│
└── data/                    # Created automatically at runtime
    └── faiss_index/
        ├── index.faiss
        └── chunks.pkl
```

The knowledge base source of truth for these documents is the AreebaCare reference document:
`https://docs.google.com/document/d/1vbCKrTsSntQvPJqaAHvr3suqy5O0QQFs7OHL93IPkP0/edit?tab=t.0`

The application itself never connects to Google Drive — export the content from that document into the PDF files listed above and place them inside `knowledge_base/` before running the app.

## 5. Installation (local machine)

```bash
git clone <your-repo-url> AreebaCare
cd AreebaCare
pip install -r requirements.txt
```

Place your PDF files inside `knowledge_base/`, then run:

```bash
streamlit run app.py
```

## 6. Google Colab Setup

1. Upload or clone the project so it lives at `/content/AreebaCare/`.
2. Install dependencies:

```python
!pip install -q -r /content/AreebaCare/requirements.txt
```

3. Place your PDF files inside `/content/AreebaCare/knowledge_base/`.
4. Set the Groq API key (see Section 7).
5. Run Streamlit with a Cloudflare Quick Tunnel (see Section 9).

## 7. GROQ_API_KEY Setup

The application **never** hardcodes the API key and never displays it anywhere in the UI. It is read only from the environment variable `GROQ_API_KEY`.

In Google Colab:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_KEY"
```

Do **not** commit your real key to GitHub or paste it into shared notebooks. Prefer a secure mechanism such as Colab's "Secrets" panel (the key icon in the left sidebar) or a local `.env` file that is excluded from version control:

```python
import os
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

If `GROQ_API_KEY` is not set, the app displays a clean warning and will not attempt to call Groq.

## 8. Running Streamlit

```bash
cd /content/AreebaCare
streamlit run app.py --server.port 8501 --server.headless true
```

## 9. Cloudflare Quick Tunnel (Google Colab)

In a separate Colab cell, after Streamlit is running:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8501
```

Cloudflared will print a public `https://<random>.trycloudflare.com` URL. Open it to access the running AreebaCare app. The application itself does not manage the tunnel — this is purely a Colab networking step.

## 10. Knowledge-Base PDF Placement

Place every AreebaCare PDF directly inside the `knowledge_base/` folder (no subfolders). The app automatically:

- Detects all `.pdf` files in that folder
- Extracts and cleans their text
- Splits the text into overlapping chunks
- Builds embeddings and a FAISS index the first time it runs

If a PDF fails to process, it is skipped and reported in the app instead of crashing the whole pipeline.

## 11. FAISS Indexing

On first run (or whenever `data/faiss_index/` is missing), the app builds the FAISS index automatically from the PDFs and saves it to disk. On subsequent runs it loads the saved index instead of rebuilding it, which keeps startup fast. Use the **Rebuild Knowledge Base** button in the app whenever you add, remove, or change PDF files.

## 12. Voice Input

Voice questions use `streamlit-mic-recorder` to capture audio directly in the browser and `faster-whisper` (model size `small`, CPU-friendly settings) to transcribe it. Once a recording is available, it is processed automatically — there is no separate "Ask" button. The transcript, text answer, and audio answer are all shown automatically.

## 13. Text-to-Speech

Every successful answer is converted to speech using `gTTS` in the selected language (English, Urdu, Roman Urdu). If speech synthesis fails for any reason, the text answer remains fully visible — audio failures never block the text response.

## 14. Safety & Limitations

- AreebaCare only answers from the local PDF knowledge base; it never falls back to general medical knowledge.
- If relevant information isn't found, it says so explicitly rather than guessing.
- It never diagnoses conditions, prescribes medication, recommends treatment, or claims to replace a doctor.
- For potentially urgent questions, it advises seeking immediate professional or emergency care rather than attempting to help directly.
- AreebaCare and its knowledge base are entirely fictional and intended for educational/demonstration purposes only.

---

**AreebaCare Healthcare Assistant • Educational Project**
Created by Areeba Imran
Designed & Developed by Areeba Imran
© 2026 Areeba Imran. All rights reserved.
