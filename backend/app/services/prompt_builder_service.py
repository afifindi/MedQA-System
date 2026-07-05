"""
Medical QA System - Prompt Builder Service.

Responsible for constructing the exact prompt passed to the Gemma generator.
Uses the agreed prompt template verbatim (as specified in the project
requirements) so that the fine-tuned LoRA adapter receives input in the same
format it was trained on.

Do NOT alter the prompt template without retraining the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Candidate column names to search for document text, in priority order.
# The knowledge base CSV may use any of these as its main text column.
# ---------------------------------------------------------------------------
_TEXT_COLUMN_CANDIDATES: List[str] = [
    "answer", "Answer",
    "text", "Text",
    "content", "Content",
    "document", "Document",
    "description", "Description",
    "context", "Context",
]


class PromptBuilderService:
    """
    Constructs prompts for the Gemma-2B-IT + LoRA medical QA model.

    The service has two responsibilities:

    1. **Context building** – convert a list of retrieved document dicts
       into a single coherent context string.
    2. **Prompt formatting** – embed the context and user question into the
       exact template the fine-tuned model expects.

    The prompt template is defined as a class constant so it is visible and
    auditable without instantiating the class.
    """

    # -----------------------------------------------------------------------
    # Prompt template – matches the training configuration exactly.
    # -----------------------------------------------------------------------
    PROMPT_TEMPLATE: str = (
        "You are a helpful medical assistant.\n"
        "Answer ONLY using the provided medical context.\n"
        "If the answer is unavailable in the context, clearly state that "
        "the information cannot be found.\n"
        "\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    # Separator between document chunks in the context string
    DOCUMENT_SEPARATOR: str = "\n\n---\n\n"

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Convert a list of knowledge-base document dicts into a context string.

        Each document is converted to a text fragment by looking up its text
        field using the priority list :data:`_TEXT_COLUMN_CANDIDATES`.  If
        none of the candidate columns are present the entire row dict is
        serialised as a string (safe fallback).

        Documents are joined with :attr:`DOCUMENT_SEPARATOR`.

        Args:
            documents: List of dicts as returned by
                       :meth:`KnowledgeBaseService.get_documents_by_indices`.

        Returns:
            A single context string ready for insertion into the prompt.
            Returns an empty string if the list is empty.
        """
        if not documents:
            return ""

        fragments: List[str] = []
        for doc in documents:
            text = self._extract_text(doc)
            if text:
                fragments.append(text.strip())

        return self.DOCUMENT_SEPARATOR.join(fragments)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, context: str, question: str) -> str:
        """
        Format the context and question into the model's prompt template.

        Args:
            context:  The context string produced by :meth:`build_context`.
            question: The validated user question string.

        Returns:
            The complete formatted prompt string ready for tokenisation.
        """
        return self.PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

    def format_with_chat_template(
        self, tokenizer: Any, prompt: str
    ) -> str:
        """
        Optionally wrap *prompt* in the tokeniser's chat template.

        If the tokeniser exposes ``apply_chat_template`` (as modern instruction-
        tuned tokenisers do) the prompt is wrapped in the standard chat message
        format, which the LoRA adapter may have been trained to expect.

        Falls back to returning *prompt* unchanged on any error.

        Args:
            tokenizer: A loaded Hugging Face tokeniser object.
            prompt:    The raw formatted prompt string.

        Returns:
            The prompt string, optionally wrapped in the chat template.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Graceful fallback – return prompt as-is
            return prompt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(document: Dict[str, Any]) -> Optional[str]:
        """
        Extract the main text value from a document dict.

        Tries column name candidates in the order defined by
        :data:`_TEXT_COLUMN_CANDIDATES`.  Falls back to str(document) if
        none of the candidates are present in the dict.

        Args:
            document: A row dict from the knowledge base.

        Returns:
            The extracted text string, or None if the document is empty.
        """
        if not document:
            return None

        for key in _TEXT_COLUMN_CANDIDATES:
            value = document.get(key)
            if value is not None and str(value).strip():
                return str(value)

        # Last resort: serialise the entire dict
        return str(document)
