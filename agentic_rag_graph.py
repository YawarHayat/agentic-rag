import logging
from typing import List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain.docstore.document import Document

from embedding_store import load_vectorstore
from agent_decisions import (
    should_use_llm_selector,
    build_chunk_preview_list,
    select_relevant_chunks,
    answer_and_check_context,
    rewrite_query,
    summarize_web_results,
    search_web,
)
from config import RETRIEVAL_TOP_K, FIXED_CHUNK_TOP_K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed state — every field the graph reads or writes is declared here
# ---------------------------------------------------------------------------

class RAGState(TypedDict, total=False):
    input: str
    allow_web_search: bool
    retry_count: int
    use_llm_selector: bool
    retrieved_chunks: List[Document]
    chunk_previews: str
    selected_chunks: List[Document]
    answer: str
    context_enough: bool
    final_answer: str
    steps: List[str]          # ordered list of agent nodes that executed


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def _append_step(state: RAGState, step: str) -> List[str]:
    return state.get("steps", []) + [step]


def retriever_node(state: RAGState) -> RAGState:
    question = state["input"]
    store = load_vectorstore()
    docs = store.similarity_search(question, k=RETRIEVAL_TOP_K)
    logger.info("[Retriever] Retrieved %d chunks.", len(docs))
    return {**state, "retrieved_chunks": docs, "steps": _append_step(state, "Retriever")}


def decider_node(state: RAGState) -> RAGState:
    use_llm = should_use_llm_selector(state["input"])
    logger.info("[Decider] Use LLM selector: %s", use_llm)
    return {**state, "use_llm_selector": use_llm, "steps": _append_step(state, "Decider")}


def chunk_preview_node(state: RAGState) -> RAGState:
    preview = build_chunk_preview_list(state["retrieved_chunks"])
    logger.info("[Chunk Preview] Preview built.")
    return {**state, "chunk_previews": preview, "steps": _append_step(state, "Chunk Preview")}


def chunk_selector_node(state: RAGState) -> RAGState:
    selected = select_relevant_chunks(state["input"], state["retrieved_chunks"])
    logger.info("[Chunk Selector] Selected %d chunks.", len(selected))
    return {**state, "selected_chunks": selected, "steps": _append_step(state, "LLM Chunk Selector")}


def fixed_chunk_node(state: RAGState) -> RAGState:
    selected = state["retrieved_chunks"][:FIXED_CHUNK_TOP_K]
    logger.info("[Fixed Chunk] Using top-%d chunks.", FIXED_CHUNK_TOP_K)
    return {**state, "selected_chunks": selected, "steps": _append_step(state, "Fixed Chunk (top-4)")}


def answer_node(state: RAGState) -> RAGState:
    answer, enough = answer_and_check_context(state["input"], state["selected_chunks"])
    logger.info("[Answer] Context sufficient: %s", enough)
    return {**state, "answer": answer, "context_enough": enough, "steps": _append_step(state, "Generate Answer")}


def rewrite_node(state: RAGState) -> RAGState:
    new_query = rewrite_query(state["input"])
    logger.info("[Rewrite] Query rewritten.")
    return {
        **state,
        "input": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": _append_step(state, "Query Rewriter"),
    }


def web_search_node(state: RAGState) -> RAGState:
    results = search_web(state["input"])
    summary = summarize_web_results(state["input"], results)
    logger.info("[Web Search] Answer generated from web.")
    return {**state, "final_answer": summary, "steps": _append_step(state, "Web Search")}


def finalize_node(state: RAGState) -> RAGState:
    final = state.get("final_answer") or state.get("answer", "No answer found.")
    logger.info("[Finalize] Pipeline complete.")
    return {**state, "final_answer": final, "steps": _append_step(state, "Finalize")}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def decider_router(state: RAGState) -> str:
    return "chunk_preview" if state.get("use_llm_selector") else "fixed_chunk"


def answer_router(state: RAGState) -> str:
    if state.get("context_enough"):
        return "finalize"

    # retry_count is incremented by rewrite_node after the first rewrite attempt
    if state.get("retry_count", 0) == 0:
        return "rewrite"

    if state.get("allow_web_search"):
        return "web_search"

    logger.info("[Router] Context insufficient, web search disabled. Finalizing.")
    return "finalize"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retriever",      retriever_node)
    graph.add_node("decider",        decider_node)
    graph.add_node("chunk_preview",  chunk_preview_node)
    graph.add_node("chunk_selector", chunk_selector_node)
    graph.add_node("fixed_chunk",    fixed_chunk_node)
    graph.add_node("generate_answer", answer_node)
    graph.add_node("rewrite",         rewrite_node)
    graph.add_node("web_search",      web_search_node)
    graph.add_node("finalize",        finalize_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "decider")
    graph.add_conditional_edges(
        "decider",
        decider_router,
        {"chunk_preview": "chunk_preview", "fixed_chunk": "fixed_chunk"},
    )
    graph.add_edge("chunk_preview",   "chunk_selector")
    graph.add_edge("chunk_selector",  "generate_answer")
    graph.add_edge("fixed_chunk",     "generate_answer")
    graph.add_conditional_edges(
        "generate_answer",
        answer_router,
        {"finalize": "finalize", "rewrite": "rewrite", "web_search": "web_search"},
    )
    graph.add_edge("rewrite",    "retriever")
    graph.add_edge("web_search", "finalize")
    graph.add_edge("finalize",   END)

    return graph


rag_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public interfaces
# ---------------------------------------------------------------------------

def run_agentic_rag(question: str, allow_web_search: bool = False) -> str:
    executor = rag_graph.compile()
    initial_state: RAGState = {
        "input": question,
        "allow_web_search": allow_web_search,
        "retry_count": 0,
    }
    final_state: RAGState = executor.invoke(initial_state)
    answer = final_state.get("final_answer", "No answer found.")
    logger.info("[run_agentic_rag] Done.")
    return answer


def run_agentic_rag_with_ui_info(question: str, allow_web_search: bool = False) -> dict:
    executor = rag_graph.compile()
    initial_state: RAGState = {
        "input": question,
        "allow_web_search": allow_web_search,
        "retry_count": 0,
    }
    final_state: RAGState = executor.invoke(initial_state)

    chunks = final_state.get("retrieved_chunks", [])
    retrieved_chunks = [
        doc.page_content for doc in chunks if hasattr(doc, "page_content")
    ]

    rewritten_input = final_state.get("input", question)
    is_rewritten = rewritten_input != question

    return {
        "original_query": question,
        "rewritten_query": rewritten_input if is_rewritten else None,
        "retrieved_chunks": retrieved_chunks,
        "context_enough": final_state.get("context_enough", False),
        "final_answer": final_state.get("final_answer", "No answer found."),
        "steps": final_state.get("steps", []),
    }
