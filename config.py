import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.1-8b-instant"

# --- Embeddings ---
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# --- Vector Store ---
CHROMA_DIR: str = "chroma_db"
CHROMA_COLLECTION: str = "rag_collection"

# --- Document Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# --- Retrieval ---
RETRIEVAL_TOP_K: int = 10
SELECTOR_TOP_K: int = 10
FIXED_CHUNK_TOP_K: int = 4

# --- Web Search ---
WEB_SEARCH_RESULTS: int = 5

# --- Logging ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
