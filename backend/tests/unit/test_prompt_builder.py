"""
Unit Tests - PromptBuilderService.

Tests the exact prompt template output, context building with different
column name variants, and fallback behaviour for empty input.
"""

from __future__ import annotations

import pytest

from app.services.prompt_builder_service import PromptBuilderService


@pytest.fixture()
def svc() -> PromptBuilderService:
    """Return a fresh PromptBuilderService instance."""
    return PromptBuilderService()


class TestBuildPrompt:
    """Tests for PromptBuilderService.build_prompt()."""

    def test_build_prompt_contains_context(self, svc: PromptBuilderService) -> None:
        """The formatted prompt must include the context string verbatim."""
        prompt = svc.build_prompt(context="Context content here.", question="Q?")
        assert "Context content here." in prompt

    def test_build_prompt_contains_question(self, svc: PromptBuilderService) -> None:
        """The formatted prompt must include the question string verbatim."""
        prompt = svc.build_prompt(context="ctx", question="What is diabetes?")
        assert "What is diabetes?" in prompt

    def test_build_prompt_uses_exact_template_structure(
        self, svc: PromptBuilderService
    ) -> None:
        """The prompt must contain the required static header text."""
        prompt = svc.build_prompt(context="ctx", question="q")
        assert "You are a helpful medical assistant." in prompt
        assert "Answer ONLY using the provided medical context." in prompt
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt

    def test_build_prompt_ends_with_answer_label(
        self, svc: PromptBuilderService
    ) -> None:
        """The prompt should end with 'Answer:' to trigger generation."""
        prompt = svc.build_prompt(context="c", question="q")
        assert prompt.rstrip().endswith("Answer:")

    def test_build_prompt_empty_context(self, svc: PromptBuilderService) -> None:
        """Empty context should still produce a valid prompt."""
        prompt = svc.build_prompt(context="", question="What is a virus?")
        assert "What is a virus?" in prompt
        assert "Context:" in prompt


class TestBuildContext:
    """Tests for PromptBuilderService.build_context()."""

    def test_build_context_tries_answer_key_first(
        self, svc: PromptBuilderService
    ) -> None:
        """The 'answer' column takes priority over other columns."""
        docs = [{"answer": "Diabetes text", "text": "Other text"}]
        context = svc.build_context(docs)
        assert "Diabetes text" in context
        assert "Other text" not in context

    def test_build_context_falls_back_to_text_key(
        self, svc: PromptBuilderService
    ) -> None:
        """Falls back to 'text' column when 'answer' is absent."""
        docs = [{"text": "Hypertension content"}]
        context = svc.build_context(docs)
        assert "Hypertension content" in context

    def test_build_context_falls_back_to_content_key(
        self, svc: PromptBuilderService
    ) -> None:
        """Falls back to 'content' column when 'answer' and 'text' are absent."""
        docs = [{"content": "Asthma info"}]
        context = svc.build_context(docs)
        assert "Asthma info" in context

    def test_build_context_joins_multiple_docs_with_separator(
        self, svc: PromptBuilderService
    ) -> None:
        """Multiple documents are joined with the defined separator."""
        docs = [{"answer": "Doc A"}, {"answer": "Doc B"}]
        context = svc.build_context(docs)
        assert "Doc A" in context
        assert "Doc B" in context
        assert svc.DOCUMENT_SEPARATOR in context

    def test_build_context_handles_empty_list(
        self, svc: PromptBuilderService
    ) -> None:
        """Empty document list returns empty string."""
        context = svc.build_context([])
        assert context == ""

    def test_build_context_serialises_unknown_columns(
        self, svc: PromptBuilderService
    ) -> None:
        """Docs without any known text column fall back to str(doc)."""
        docs = [{"completely_unknown_col": "value"}]
        context = svc.build_context(docs)
        # Should not be empty; str(doc) is used as fallback
        assert context != ""
