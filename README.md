# Agentic RAG — Document Question-Answering System

A multi-agent Retrieval-Augmented Generation (RAG) system built as my final-year thesis project. The core idea was to go beyond a naive "retrieve and answer" pipeline and design a system where separate, specialised agents make independent decisions at each stage — deciding *how* to retrieve, *which* chunks are relevant, *whether* the answer is sufficient, and *when* to fall back to the web.

---

## System Architecture

The pipeline is implemented as a directed graph using [LangGraph](https://github.com/langchain-ai/langgraph), where each node is an agent with a single responsibility.

```
                          ┌─────────────┐
                          │   Retriever │  similarity search (top-10)
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   Decider   │  heuristic: simple vs. complex query
                          └──────┬──────┘
                   ┌─────────────┴─────────────┐
           complex │                           │ simple
    ┌──────▼──────┐                   ┌────────▼────────┐
    │Chunk Preview│                   │  Fixed Chunk    │  top-4
    └──────┬──────┘                   └────────┬────────┘
           │                                   │
    ┌──────▼──────┐                            │
    │LLM Selector │  LLM picks best chunks     │
    └──────┬──────┘                            │
           └─────────────┬─────────────────────┘
                  ┌──────▼──────┐
                  │   Answer    │  generate answer + sufficiency check
                  └──────┬──────┘
          ┌───────────────┼───────────────┐
    yes   │          no (retry 0)         │ no (retry 1+)
  ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
  │   Finalize  │ │   Rewrite   │ │ Web Search │
  └─────────────┘ └──────┬──────┘ └─────┬──────┘
                         │               │
                         └───────┬───────┘
                          ┌──────▼──────┐
                          │   Finalize  │
                          └─────────────┘
```

### Agent Responsibilities

| Agent | Role |
|---|---|
| **Retriever** | Embeds the query and performs cosine similarity search against the ChromaDB vector store |
| **Decider** | Applies a lightweight heuristic to determine whether the query needs LLM-based chunk selection or a fixed top-k approach |
| **Chunk Preview** | Builds truncated previews of retrieved chunks to fit within the LLM context window |
| **LLM Chunk Selector** | Sends chunk previews to the LLM and asks it to return only the most relevant chunk indices |
| **Fixed Chunk** | Falls back to the top-4 retrieved chunks without calling the LLM — faster for simple queries |
| **Answer Generator** | Constructs a grounded answer using selected chunks and evaluates whether the context was sufficient |
| **Query Rewriter** | If context is insufficient on the first attempt, rewrites the query for better retrieval without injecting external knowledge |
| **Web Search** | Falls back to DuckDuckGo when document context remains insufficient after a rewrite |
| **Finalize** | Consolidates the final answer from whichever path produced it |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (`StateGraph`) |
| LLM (reasoning & answers) | Groq API — LLaMA 3.3 70B |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (local persistence) |
| Document parsing | PyPDF2, python-docx |
| Web search fallback | DuckDuckGo Search |
| UI | Streamlit |
| Containerisation | Docker |

---

## Project Structure

```
.
├── config.py              # Centralised configuration (models, paths, chunk sizes)
├── document_loader.py     # PDF / DOCX ingestion and chunking
├── embedding_store.py     # ChromaDB store / load / reset
├── agent_decisions.py     # All LLM-backed agent logic
├── agentic_rag_graph.py   # LangGraph graph definition and compiled pipeline
├── streamlit_app.py       # Streamlit web UI
├── main.py                # CLI entry point (ingest / query / run)
├── render_graph.py        # Utility to export the graph as a PNG
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YawarHayat/agentic-rag.git
cd agentic-rag
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your [Groq API key](https://console.groq.com):

```
GROQ_API_KEY=your_key_here
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the web UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, upload a PDF or DOCX, and start asking questions.

### 5. Or use the CLI

```bash
# Ingest a document
python main.py ingest --file ./documents/report.pdf

# Ask a question
python main.py query --question "What is the main conclusion?"

# Ingest and query in one step
python main.py run --file ./documents/report.pdf --question "Summarise chapter 3" --web
```

---

## Docker

```bash
docker build -t agentic-rag .
docker run -p 8501:8501 --env-file .env agentic-rag
```

---

## Design Decisions

**Why LangGraph instead of a simple chain?**
The retry-and-rewrite loop requires a cycle in the execution graph. LangGraph's `StateGraph` supports conditional edges and cycles natively, which made it the right fit over a linear LangChain chain.

**Why a separate Decider agent?**
For short, factual queries the overhead of sending chunk previews to the LLM for selection is unnecessary. The Decider uses a fast heuristic (word count + keyword matching) to route simple queries directly to fixed-top-k selection, reducing latency and token usage.

**Why `all-MiniLM-L6-v2` for embeddings?**
It runs entirely locally with no API cost, loads in under a second, and performs well on general semantic similarity tasks. For a thesis prototype this was the right balance between quality and simplicity.

**Why Groq + LLaMA 3.3 70B?**
The LLaMA 3.3 70B model on Groq's inference API delivers strong reasoning quality at speeds fast enough for interactive use. For a system that may call the LLM three or four times per query (selection, answer, rewrite, summary), both quality and latency matter — Groq's hardware gives both.
