import os
import logging
import streamlit as st

from document_loader import load_and_split_document
from embedding_store import store_embeddings, reset_chroma_db
from agentic_rag_graph import run_agentic_rag_with_ui_info
from config import LOG_LEVEL, GROQ_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — subtle professional styling, not flashy
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem; }

    /* Step badge */
    .step-badge {
        display: inline-block;
        background: #1f4e79;
        color: #ffffff;
        border-radius: 12px;
        padding: 3px 12px;
        margin: 3px 4px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    /* Section headers */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 6px;
    }

    /* Answer box */
    .answer-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 0.96rem;
        line-height: 1.6;
        color: #111827;
    }

    /* Warning box */
    .warn-box {
        background: #fffbeb;
        border-left: 4px solid #d97706;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #92400e;
    }

    /* History item */
    .history-q { font-weight: 600; color: #1e3a5f; margin-bottom: 2px; }
    .history-a { color: #374151; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    save_dir = "documents"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def render_step_trace(steps: list[str]) -> None:
    badges = "".join(f'<span class="step-badge">{s}</span>' for s in steps)
    st.markdown(
        f'<p class="section-label">Pipeline trace</p>{badges}',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Agentic RAG")
    st.caption("Document Question-Answering System")
    st.divider()

    st.subheader("Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX file",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Search Settings")
    allow_web = st.toggle("Allow web search", value=False,
                          help="Falls back to DuckDuckGo when document context is insufficient")
    chunk_size_display = CHUNK_SIZE
    overlap_display = CHUNK_OVERLAP

    st.divider()
    st.subheader("Model Info")
    st.caption(f"LLM: `{GROQ_MODEL}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption(f"Chunk size: `{chunk_size_display}` / overlap: `{overlap_display}`")

    st.divider()
    if st.button("Clear conversation history"):
        st.session_state.history.clear()
        st.success("History cleared.")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.header("Document Q&A")

if uploaded_file is None:
    st.info("Upload a PDF or DOCX file in the sidebar to get started.")
    st.stop()

# --- Ingest document if new ---
if st.session_state.processed_file != uploaded_file.name:
    with st.status("Processing document…", expanded=True) as status:
        st.write("Clearing previous vector store…")
        reset_chroma_db()
        st.session_state.history.clear()
        st.session_state.processed_file = uploaded_file.name

        st.write("Loading and chunking document…")
        file_path = save_uploaded_file(uploaded_file)
        chunks = load_and_split_document(file_path)

        st.write(f"Storing {len(chunks)} chunks in ChromaDB…")
        store_embeddings(chunks)
        status.update(label=f"Ready — {len(chunks)} chunks indexed from **{uploaded_file.name}**",
                      state="complete")

    with st.expander("Preview first chunk"):
        st.code(chunks[0].page_content[:600], language=None)

st.divider()

# --- Question input ---
question = st.text_input(
    "Ask a question about the document",
    placeholder="e.g. What is the main argument of chapter 2?",
)

col_submit, col_space = st.columns([1, 5])
with col_submit:
    submitted = st.button("Ask", type="primary", use_container_width=True)

if submitted and question.strip():
    with st.spinner("Running pipeline…"):
        response = run_agentic_rag_with_ui_info(question.strip(), allow_web_search=allow_web)

    st.session_state.history.append((question.strip(), response))

    # --- Pipeline trace ---
    if response.get("steps"):
        render_step_trace(response["steps"])
        st.write("")

    # --- Rewritten query notice ---
    if response.get("rewritten_query"):
        st.markdown(
            '<p class="section-label">Query rewritten for clarity</p>',
            unsafe_allow_html=True,
        )
        st.code(response["rewritten_query"], language=None)

    # --- Context warning ---
    if not response["context_enough"]:
        msg = (
            "Document context was insufficient. Answer is based on web search results."
            if allow_web
            else "Document context was insufficient. Web search was disabled — answer may be incomplete."
        )
        st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)
        st.write("")

    # --- Answer ---
    st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{response["final_answer"]}</div>',
        unsafe_allow_html=True,
    )

    # --- Retrieved chunks ---
    if response["retrieved_chunks"]:
        st.write("")
        with st.expander(f"Retrieved chunks ({len(response['retrieved_chunks'])})"):
            for i, chunk in enumerate(response["retrieved_chunks"], 1):
                st.markdown(f"**Chunk {i}**")
                st.code(chunk[:500], language=None)
                st.write("")

elif submitted:
    st.warning("Please enter a question before submitting.")

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("Conversation History")
    for i, (q, res) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f'<p class="history-q">Q{i}: {q}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="history-a">{res["final_answer"]}</p>', unsafe_allow_html=True)
