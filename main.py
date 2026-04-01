"""
main.py — CLI entry point for the Agentic RAG pipeline.

Usage:
    python main.py --file ./documents/my_doc.pdf --question "What is the main theme?"
    python main.py --file ./documents/my_doc.pdf --question "Summarize chapter 1" --web
"""

import argparse
import logging

from document_loader import load_and_split_document
from embedding_store import store_embeddings
from agentic_rag_graph import run_agentic_rag
from config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingest(file_path: str) -> None:
    logger.info("Loading and chunking document: %s", file_path)
    chunks = load_and_split_document(file_path)
    logger.info("Created %d chunks.", len(chunks))
    store_embeddings(chunks)
    logger.info("Embeddings stored successfully.")


def query(question: str, allow_web: bool) -> str:
    logger.info("Running pipeline for question: %s", question)
    answer = run_agentic_rag(question, allow_web_search=allow_web)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG — document question-answering pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Load a document into the vector store")
    ingest_parser.add_argument("--file", required=True, help="Path to PDF or DOCX file")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Ask a question against stored documents")
    query_parser.add_argument("--question", required=True, help="Question to answer")
    query_parser.add_argument(
        "--web", action="store_true", default=False,
        help="Allow web search when document context is insufficient"
    )

    # Ingest + query in one step
    run_parser = subparsers.add_parser("run", help="Ingest a document then ask a question")
    run_parser.add_argument("--file", required=True, help="Path to PDF or DOCX file")
    run_parser.add_argument("--question", required=True, help="Question to answer")
    run_parser.add_argument("--web", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.file)

    elif args.command == "query":
        answer = query(args.question, args.web)
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        print(answer)

    elif args.command == "run":
        ingest(args.file)
        answer = query(args.question, args.web)
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()
