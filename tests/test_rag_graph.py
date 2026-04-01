"""
Integration tests for the RAG graph pipeline.

The entire LLM and vector store layer is mocked so tests run
instantly without API keys, GPU, or a populated ChromaDB.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.docstore.document import Document

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_docs(n=5):
    return [Document(page_content=f"Context chunk {i} about the mystery.") for i in range(n)]


def _mock_groq_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = text
    return mock


# ---------------------------------------------------------------------------
# answer_router logic
# ---------------------------------------------------------------------------

class TestAnswerRouter:
    def test_routes_to_finalize_when_context_enough(self):
        from agentic_rag_graph import answer_router
        state = {"context_enough": True, "retry_count": 0, "allow_web_search": False}
        assert answer_router(state) == "finalize"

    def test_routes_to_rewrite_on_first_failure(self):
        from agentic_rag_graph import answer_router
        state = {"context_enough": False, "retry_count": 0, "allow_web_search": False}
        assert answer_router(state) == "rewrite"

    def test_routes_to_web_search_on_second_failure_with_web_allowed(self):
        from agentic_rag_graph import answer_router
        state = {"context_enough": False, "retry_count": 1, "allow_web_search": True}
        assert answer_router(state) == "web_search"

    def test_routes_to_finalize_on_second_failure_without_web(self):
        from agentic_rag_graph import answer_router
        state = {"context_enough": False, "retry_count": 1, "allow_web_search": False}
        assert answer_router(state) == "finalize"


# ---------------------------------------------------------------------------
# decider_router logic
# ---------------------------------------------------------------------------

class TestDeciderRouter:
    def test_routes_to_chunk_preview_when_llm_selector_true(self):
        from agentic_rag_graph import decider_router
        state = {"use_llm_selector": True}
        assert decider_router(state) == "chunk_preview"

    def test_routes_to_fixed_chunk_when_llm_selector_false(self):
        from agentic_rag_graph import decider_router
        state = {"use_llm_selector": False}
        assert decider_router(state) == "fixed_chunk"


# ---------------------------------------------------------------------------
# Full pipeline — happy path (context sufficient on first try)
# ---------------------------------------------------------------------------

class TestFullPipelineHappyPath:
    def test_returns_answer_when_context_sufficient(self, mocker):
        from agentic_rag_graph import run_agentic_rag_with_ui_info

        docs = _make_docs()
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = docs
        mocker.patch("agentic_rag_graph.load_vectorstore", return_value=mock_store)

        # Short query (< 10 words, no complex keywords) → fixed_chunk path
        # Only 1 LLM call: answer_and_check_context
        mocker.patch(
            "agent_decisions.groq_client.chat.completions.create",
            side_effect=[
                _mock_groq_response("Mr. Curtis was poisoned.\nContext sufficient: Yes"),
            ],
        )

        result = run_agentic_rag_with_ui_info("Who killed Curtis?", allow_web_search=False)

        assert result["final_answer"] == "Mr. Curtis was poisoned."
        assert result["context_enough"] is True
        assert result["rewritten_query"] is None
        assert "Retriever" in result["steps"]
        assert "Finalize" in result["steps"]

    def test_steps_list_is_ordered_correctly(self, mocker):
        from agentic_rag_graph import run_agentic_rag_with_ui_info

        docs = _make_docs()
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = docs
        mocker.patch("agentic_rag_graph.load_vectorstore", return_value=mock_store)

        # Short query → fixed_chunk path → 1 LLM call only
        mocker.patch(
            "agent_decisions.groq_client.chat.completions.create",
            side_effect=[
                _mock_groq_response("The answer.\nContext sufficient: Yes"),
            ],
        )

        result = run_agentic_rag_with_ui_info("short query?")
        steps = result["steps"]

        # Retriever must come before Finalize
        assert steps.index("Retriever") < steps.index("Finalize")


# ---------------------------------------------------------------------------
# Full pipeline — rewrite path (context insufficient → rewrite → sufficient)
# ---------------------------------------------------------------------------

class TestFullPipelineRewritePath:
    def test_rewrite_path_sets_rewritten_query(self, mocker):
        from agentic_rag_graph import run_agentic_rag_with_ui_info

        docs = _make_docs()
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = docs
        mocker.patch("agentic_rag_graph.load_vectorstore", return_value=mock_store)

        mocker.patch(
            "agent_decisions.groq_client.chat.completions.create",
            side_effect=[
                # First retrieval: fixed chunk path (short query) → answer → insufficient
                _mock_groq_response("No information found.\nContext sufficient: No"),
                # rewrite_query call
                _mock_groq_response("What happened to Mr. Curtis in the story?"),
                # Second retrieval: fixed chunk → answer → sufficient
                _mock_groq_response("Mr. Curtis was murdered.\nContext sufficient: Yes"),
            ],
        )

        result = run_agentic_rag_with_ui_info("Curtis?", allow_web_search=False)

        assert result["final_answer"] == "Mr. Curtis was murdered."
        assert result["rewritten_query"] is not None
        assert "Query Rewriter" in result["steps"]


# ---------------------------------------------------------------------------
# Web search fallback path
# ---------------------------------------------------------------------------

class TestWebSearchFallback:
    def test_web_search_used_when_context_insufficient_twice(self, mocker):
        from agentic_rag_graph import run_agentic_rag_with_ui_info

        docs = _make_docs()
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = docs
        mocker.patch("agentic_rag_graph.load_vectorstore", return_value=mock_store)

        mocker.patch(
            "agent_decisions.groq_client.chat.completions.create",
            side_effect=[
                # First answer attempt — insufficient
                _mock_groq_response("Not enough info.\nContext sufficient: No"),
                # rewrite_query
                _mock_groq_response("What was the motive for the crime?"),
                # Second answer attempt — still insufficient
                _mock_groq_response("Still not enough.\nContext sufficient: No"),
                # web search summary
                _mock_groq_response("According to web results, the motive was revenge."),
            ],
        )

        # Mock web search itself
        mocker.patch(
            "agent_decisions.search_web",
            return_value="Web snippet: motive was revenge (example.com)",
        )

        result = run_agentic_rag_with_ui_info("motive?", allow_web_search=True)

        assert "Web Search" in result["steps"]
        assert "revenge" in result["final_answer"].lower()
