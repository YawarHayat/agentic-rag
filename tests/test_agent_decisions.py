"""
Tests for agent_decisions.py

These tests cover the pure-logic and LLM-backed functions using mocking
so they run fast without real API calls or network access.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.docstore.document import Document

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# should_use_llm_selector
# ---------------------------------------------------------------------------

class TestShouldUseLLMSelector:
    def test_short_simple_query_returns_false(self):
        from agent_decisions import should_use_llm_selector
        assert should_use_llm_selector("Who is Daisy?") is False

    def test_long_query_returns_true(self):
        from agent_decisions import should_use_llm_selector
        # More than 10 words
        assert should_use_llm_selector(
            "Can you tell me what happens in the first chapter of the book?"
        ) is True

    def test_complex_keyword_summarize_returns_true(self):
        from agent_decisions import should_use_llm_selector
        assert should_use_llm_selector("summarize the plot") is True

    def test_complex_keyword_compare_returns_true(self):
        from agent_decisions import should_use_llm_selector
        assert should_use_llm_selector("compare the two characters") is True

    def test_complex_keyword_explain_returns_true(self):
        from agent_decisions import should_use_llm_selector
        assert should_use_llm_selector("explain the motive") is True


# ---------------------------------------------------------------------------
# build_chunk_preview_list
# ---------------------------------------------------------------------------

class TestBuildChunkPreviewList:
    def test_returns_numbered_list(self):
        from agent_decisions import build_chunk_preview_list
        chunks = [
            Document(page_content="First chunk content here"),
            Document(page_content="Second chunk content here"),
        ]
        result = build_chunk_preview_list(chunks)
        assert result.startswith("1.")
        assert "2." in result

    def test_truncates_to_200_chars(self):
        from agent_decisions import build_chunk_preview_list
        long_text = "x" * 500
        chunks = [Document(page_content=long_text)]
        result = build_chunk_preview_list(chunks)
        # Preview is 200 chars + "1. " prefix
        assert len(result) <= 204

    def test_empty_chunks_returns_empty_string(self):
        from agent_decisions import build_chunk_preview_list
        assert build_chunk_preview_list([]) == ""

    def test_strips_newlines_in_preview(self):
        from agent_decisions import build_chunk_preview_list
        chunks = [Document(page_content="line one\nline two\nline three")]
        result = build_chunk_preview_list(chunks)
        assert "\n" not in result.split("1. ")[1]


# ---------------------------------------------------------------------------
# select_relevant_chunks  (mocked LLM)
# ---------------------------------------------------------------------------

class TestSelectRelevantChunks:
    def _make_chunks(self, n=5):
        return [Document(page_content=f"Chunk number {i}") for i in range(1, n + 1)]

    def test_returns_correct_chunks_by_index(self, mocker):
        from agent_decisions import select_relevant_chunks

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "1, 3"
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        chunks = self._make_chunks(5)
        result = select_relevant_chunks("test question", chunks)
        assert result == [chunks[0], chunks[2]]

    def test_ignores_out_of_range_indexes(self, mocker):
        from agent_decisions import select_relevant_chunks

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "1, 99"
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        chunks = self._make_chunks(3)
        result = select_relevant_chunks("test question", chunks)
        assert result == [chunks[0]]

    def test_ignores_non_numeric_tokens(self, mocker):
        from agent_decisions import select_relevant_chunks

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "1, abc, 2"
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        chunks = self._make_chunks(3)
        result = select_relevant_chunks("test question", chunks)
        assert result == [chunks[0], chunks[1]]


# ---------------------------------------------------------------------------
# answer_and_check_context  (mocked LLM)
# ---------------------------------------------------------------------------

class TestAnswerAndCheckContext:
    def _make_chunks(self, text="Some relevant context."):
        return [Document(page_content=text)]

    def test_context_sufficient_yes(self, mocker):
        from agent_decisions import answer_and_check_context

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "Mr. Curtis was poisoned.\nContext sufficient: Yes"
        )
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        answer, sufficient = answer_and_check_context("Who died?", self._make_chunks())
        assert sufficient is True
        assert "Mr. Curtis" in answer

    def test_context_sufficient_no(self, mocker):
        from agent_decisions import answer_and_check_context

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "The answer is not available.\nContext sufficient: No"
        )
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        answer, sufficient = answer_and_check_context("Who did it?", self._make_chunks())
        assert sufficient is False
        assert "Context sufficient" not in answer

    def test_answer_strips_context_line(self, mocker):
        from agent_decisions import answer_and_check_context

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "Daisy is the detective.\nContext sufficient: Yes"
        )
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        answer, _ = answer_and_check_context("Who is Daisy?", self._make_chunks())
        assert "Context sufficient" not in answer
        assert answer == "Daisy is the detective."


# ---------------------------------------------------------------------------
# rewrite_query  (mocked LLM)
# ---------------------------------------------------------------------------

class TestRewriteQuery:
    def test_returns_rewritten_string(self, mocker):
        from agent_decisions import rewrite_query

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "What is the main theme of the story?"
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        result = rewrite_query("theme?")
        assert result == "What is the main theme of the story?"

    def test_strips_whitespace(self, mocker):
        from agent_decisions import rewrite_query

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Who is the killer?  "
        mocker.patch("agent_decisions.groq_client.chat.completions.create",
                     return_value=mock_response)

        result = rewrite_query("killer")
        assert result == "Who is the killer?"
