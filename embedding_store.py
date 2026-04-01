import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def store_embeddings(docs, collection_name: str = CHROMA_COLLECTION) -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )
    logger.info("Stored %d chunks in ChromaDB collection '%s'.", len(docs), collection_name)
    return vectorstore


def load_vectorstore(collection_name: str = CHROMA_COLLECTION) -> Chroma:
    logger.info("Loading vectorstore from '%s'.", CHROMA_DIR)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )


def reset_chroma_db(collection_name: str = CHROMA_COLLECTION) -> None:
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )
        store.delete_collection()
        logger.info("ChromaDB collection '%s' deleted successfully.", collection_name)
    except Exception:
        logger.exception("Failed to reset ChromaDB collection '%s'.", collection_name)
