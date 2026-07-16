# app.py
"""
Streamlit RAG Q&A — Gemini embeddings + ChromaDB + Gemini generation.

Requirements:
 - Set GEMINI_API_KEY as an environment variable / Streamlit secret.

How to run:
 1. pip install -r requirements.txt
 2. streamlit run app.py
"""
import os
import uuid
import textwrap
from typing import List, Tuple
import sys

# --- CHROMA DB / SQLITE FIX FOR STREAMLIT CLOUD ---
# This acts as a patch to ensure ChromaDB finds a compatible SQLite version
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --------------------------------------------------

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Google GenAI SDK
from google import genai
from google.genai import types

# Chroma
import chromadb

# -------------------------
# Config
# -------------------------
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "rag_pages"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBEDDING_MODEL = "gemini-embedding-001"   # gemini embedding model
GEN_MODEL = "gemini-flash-latest"         # auto-tracks Google's current stable Flash model

# -------------------------
# UI styling
# -------------------------
st.set_page_config(page_title="RAG URL Q&A — Nadeem Gohar", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Page container padding */
    .main > .block-container { padding: 2.5rem 3.5rem; max-width: 1200px; }

    /* Deep space gradient background */
    .stApp {
        background: radial-gradient(circle at 15% 10%, #1a1c3d 0%, #0d0e21 45%, #05060f 100%);
        color: #e8e9f5;
    }

    /* Hero header block */
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(124,92,255,0.18), rgba(56,189,248,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero .byline {
        font-size: 0.95rem;
        color: #9ca3d4;
        margin-top: 0.25rem;
    }
    .hero .badge {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    /* Section labels */
    .stApp h2, .stApp h3 { font-family: 'Space Grotesk', sans-serif; color: #e8e9f5; }

    /* Markdown, text, input fields */
    .stMarkdown, .stText { color: #cfd2ec !important; }
    .stTextInput>div>div>input, .stTextArea textarea {
        background-color: rgba(255,255,255,0.05) !important;
        color: #e8e9f5 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
    }
    .stTextInput>div>div>input::placeholder, .stTextArea textarea::placeholder { color: #6b6f9c !important; }

    /* Buttons: gradient pill */
    .stButton>button, button[kind] {
        color: #ffffff !important;
        background: linear-gradient(90deg, #7c5cff, #38bdf8) !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        box-shadow: 0 4px 18px rgba(124,92,255,0.35) !important;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 6px 22px rgba(56,189,248,0.4) !important; }

    /* Panel/column wrapper feel */
    div[data-testid="column"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
    }

    /* Card style for retrieved chunks */
    .card {
        background: rgba(255,255,255,0.04);
        color: #dfe1f7;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 3px solid #38bdf8;
        margin-bottom: 0.6rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }

    /* Muted text style */
    .muted { color: #9095c4; }

    /* Alerts */
    div[data-testid="stAlert"] { border-radius: 12px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Utilities
# -------------------------
def fetch_url_text(url: str, max_chars: int = 200_000) -> Tuple[str, str]:
    headers = {"User-Agent": "RAG-Streamlit-App/1.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    body = soup.find("main") or soup.find("article")
    if body:
        texts = body.get_text(separator="\n")
    else:
        ps = soup.find_all("p")
        if ps:
            texts = "\n\n".join(p.get_text() for p in ps)
        else:
            texts = soup.get_text(separator="\n")
    title_tag = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).netloc
    texts = texts.strip()
    if len(texts) > max_chars:
        texts = texts[:max_chars]
    return title_tag, texts

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(0, end - overlap)
    return chunks

# -------------------------
# GenAI client (singleton)
# -------------------------
@st.cache_resource(show_spinner=False)
def init_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Try to get from Streamlit secrets if env var is missing
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            raise RuntimeError("GEMINI_API_KEY not found in Environment or Secrets.")
    
    client = genai.Client(api_key=api_key)
    return client

# -------------------------
# Chroma client / collection
# -------------------------
@st.cache_resource(show_spinner=False)
def init_chroma(persist_directory: str = CHROMA_DIR):
    # Updated to modern ChromaDB syntax (PersistentClient)
    client = chromadb.PersistentClient(path=persist_directory)
    return client

@st.cache_resource(show_spinner=False)
def get_or_create_collection(_chroma_client, name=CHROMA_COLLECTION_NAME):
    # Using _chroma_client to tell Streamlit not to hash this object
    return _chroma_client.get_or_create_collection(name=name)

# -------------------------
# Embedding helpers using GenAI
# -------------------------
def embed_texts(client: genai.Client, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Call Gemini embedding API for a list of texts."""
    # Modern syntax for google-genai SDK
    resp = client.models.embed_content(model=model, contents=texts)
    
    embeddings = []
    # The new SDK usually returns a list of Embedding objects directly in 'embeddings'
    if hasattr(resp, "embeddings"):
        for e in resp.embeddings:
            if hasattr(e, "values"):
                embeddings.append(e.values)
            elif isinstance(e, list):
                embeddings.append(e)
            else:
                # Fallback if structure varies
                embeddings.append(e)
    return embeddings

def embed_single(client: genai.Client, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    result = embed_texts(client, [text], model=model)
    if result:
        return result[0]
    return []

# -------------------------
# Chroma helpers (add & retrieve)
# -------------------------
def upsert_page_to_chroma(collection, url: str, title: str, chunks: List[str], embeddings: List[List[float]]):
    ids = [f"{uuid.uuid4()}_{i}" for i in range(len(chunks))]
    metadatas = [{"url": url, "title": title, "chunk_index": i} for i in range(len(chunks))]
    documents = [textwrap.shorten(c, width=2000, placeholder="") for c in chunks]
    
    # Ensure embeddings is a list of lists of floats
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

def retrieve_relevant_chunks(collection, query_embedding: List[float], url: str, top_k: int = TOP_K):
    if not query_embedding:
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"url": url},
        include=["documents", "metadatas", "distances"],
    )
    
    if not results["documents"]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    
    hits = [{"doc": d, "meta": m, "dist": dist} for d, m, dist in zip(docs, metas, dists)]
    return hits

# -------------------------
# Gemini generation
# -------------------------
def genai_generate(client: genai.Client, prompt: str, model: str = GEN_MODEL) -> str:
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=800
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    return resp.text

# -------------------------
# Prompt builder
# -------------------------
def build_rag_prompt(question: str, retrieved: List[dict]) -> str:
    context_blocks = []
    for i, hit in enumerate(retrieved):
        meta = hit.get("meta", {})
        url = meta.get("url", "unknown")
        title = meta.get("title", "")
        doc = hit.get("doc", "")
        source_tag = f""
        context_blocks.append(f"{source_tag} TITLE: {title}\nURL: {url}\nCONTENT:\n{doc}\n")
    
    context_text = "\n---\n".join(context_blocks)
    prompt = (
        "You are an expert assistant. Use ONLY the information in the provided CONTEXT to answer the user's question. "
        "If the context does not contain an answer, say so. Provide a clear, well-structured answer, "
        "followed by a short 'Sources' section listing which sources you used (by source tag) and the URL.\n\n"
        f"QUESTION: {question}\n\n"
        "CONTEXT:\n"
        f"{context_text}\n\n"
        "INSTRUCTIONS:\n"
        " - Provide a full explanation.\n"
        " - After the explanation, include a 'Sources' section.\n"
    )
    return prompt

# -------------------------
# Streamlit layout
# -------------------------
st.markdown(
    """
    <div class="hero">
        <div>
            <h1>🧠 RAG URL Q&A</h1>
            <div class="byline">Built by Nadeem Gohar · retrieval-augmented Q&A over any web page</div>
        </div>
        <div class="badge">Gemini</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("<div class='muted'>Paste a URL, ask a question about that page, and get a full explanation with citations.</div>", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 🔗 Source")
    url_input = st.text_input("Page URL", placeholder="https://example.com/article-or-doc", key="url_input")
    load_btn = st.button("Fetch & Index URL", key="index_btn")
    status_area = st.empty()

with col2:
    st.markdown("#### 💬 Ask")
    user_question = st.text_area("Your question about the page", height=140, placeholder="e.g., Summarize the main argument.")
    answer_btn = st.button("Ask AI", key="ask_btn")

# Initialize clients
try:
    genai_client = init_genai_client()
    chroma_client = init_chroma(CHROMA_DIR)
    collection = get_or_create_collection(chroma_client, CHROMA_COLLECTION_NAME)
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# Indexing flow
if load_btn:
    if not url_input:
        st.warning("Please enter a URL first.")
    else:
        status_area.info("Fetching URL...")
        try:
            title, page_text = fetch_url_text(url_input)
            status_area.success(f"Fetched: {title[:120]}")
            chunks = chunk_text(page_text)
            status_area.info(f"Text split into {len(chunks)} chunks. Generating embeddings...")
            
            embeddings = embed_texts(genai_client, chunks, model=EMBEDDING_MODEL)
            
            status_area.info("Upserting into ChromaDB...")
            upsert_page_to_chroma(collection, url_input, title, chunks, embeddings)
            
            status_area.success("Indexed successfully! ✅")
        except Exception as e:
            st.error(f"Error fetching or indexing URL: {e}")

# Asking flow
if answer_btn:
    if not user_question or not url_input:
        st.warning("Provide both a URL (indexed) and a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                q_emb = embed_single(genai_client, user_question, model=EMBEDDING_MODEL)
                hits = retrieve_relevant_chunks(collection, q_emb, url=url_input, top_k=TOP_K)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                hits = []

            if not hits:
                st.info("No relevant content found.")
            else:
                prompt = build_rag_prompt(user_question, hits)
                
                st.markdown("### 📚 Retrieved context")
                for i, h in enumerate(hits):
                    st.write(f"<div class='card'><b>Source [{i+1}]</b>: {h['doc'][:300]}...</div>", unsafe_allow_html=True)
                st.markdown("---")
                
                try:
                    answer_text = genai_generate(genai_client, prompt)
                    st.markdown("## ✨ AI Answer")
                    st.write(answer_text)
                except Exception as e:
                    st.error(f"Generation error: {e}")
