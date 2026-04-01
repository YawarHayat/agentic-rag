import logging
import os
from typing import List, Tuple

from groq import Groq
from langchain.docstore.document import Document
from duckduckgo_search import DDGS

from config import GROQ_API_KEY, GROQ_MODEL, SELECTOR_TOP_K, WEB_SEARCH_RESULTS

logger = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Heuristic: decide whether to use LLM-based chunk selection
# ---------------------------------------------------------------------------

def should_use_llm_selector(question: str) -> bool:
    question_lower = question.lower()
    if len(question.split()) > 10:
        return True
    complex_keywords = {"summarize", "compare", "explain", "analyze", "relationship", "describe"}
    return bool(complex_keywords.intersection(question_lower.split()))


# ---------------------------------------------------------------------------
# Chunk preview builder
# ---------------------------------------------------------------------------

def build_chunk_preview_list(chunks: List[Document]) -> str:
    previews = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.page_content.strip().replace("\n", " ")[:200]
        previews.append(f"{i}. {text}")
    return "\n".join(previews)


# ---------------------------------------------------------------------------
# LLM-based chunk selection
# ---------------------------------------------------------------------------

def select_relevant_chunks(
    question: str,
    chunks: List[Document],
    top_k: int = SELECTOR_TOP_K,
) -> List[Document]:
    chunk_preview = build_chunk_preview_list(chunks[:top_k])
    prompt = (
        f"You are an intelligent assistant. Given a question and {top_k} document chunks, "
        f"select the most relevant chunks.\n\n"
        f"Question:\n{question}\n\n"
        f"Chunks:\n{chunk_preview}\n\n"
        f"Return only the chunk numbers as a comma-separated list (e.g., 1,3,5), with no explanation."
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    selected_nums = response.choices[0].message.content.strip()

    indexes = []
    for part in selected_nums.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(chunks):
                indexes.append(idx)
        except ValueError:
            continue

    logger.debug("LLM selected chunk indexes: %s", indexes)
    return [chunks[i] for i in indexes]


# ---------------------------------------------------------------------------
# Answer generation with context sufficiency check
# ---------------------------------------------------------------------------

def answer_and_check_context(
    question: str,
    selected_chunks: List[Document],
) -> Tuple[str, bool]:
    context = "\n".join([doc.page_content for doc in selected_chunks])

    prompt = (
        "You are an expert analyst. Answer the question using only the provided context. "
        "Do not use external knowledge.\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        "Answer using only the information in the context. Include all relevant facts and details.\n\n"
        "Then on a new line write exactly one of:\n"
        "Context sufficient: Yes\n"
        "Context sufficient: No"
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    answer_lines = []
    context_enough = False
    for line in raw.split("\n"):
        if "context sufficient" in line.lower():
            context_enough = "yes" in line.lower()
        else:
            answer_lines.append(line.strip())

    answer = "\n".join(answer_lines).strip()
    logger.debug("Context sufficient: %s", context_enough)
    return answer, context_enough


# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------

def rewrite_query(question: str) -> str:
    prompt = (
        "The user asked the following question:\n\n"
        f"{question}\n\n"
        "This question may be unclear or too broad. Rewrite it using only the original wording — "
        "clarify or focus it, but do not inject external knowledge.\n\n"
        "Output only the rewritten question, no explanation."
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    rewritten = response.choices[0].message.content.strip()
    logger.debug("Query rewritten: '%s' -> '%s'", question, rewritten)
    return rewritten


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def search_web(query: str, num_results: int = WEB_SEARCH_RESULTS) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append(f"{r['title']}: {r['body']} ({r['href']})")
    logger.debug("Web search returned %d results.", len(results))
    return "\n\n".join(results)


def summarize_web_results(question: str, search_results: str) -> str:
    prompt = (
        "Summarize the following search results to answer the user's question.\n\n"
        f"Question: {question}\n\n"
        f"Search Results:\n{search_results}\n\n"
        "Provide a concise and accurate answer based on the search results."
    )

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content.strip()
