"""
Medical QA System - Inference Service.

Orchestrates the full RAG (Retrieval-Augmented Generation) pipeline:

  1. Validate and sanitize the user question.
  2. Embed the question using SentenceTransformers.
  3. Search the FAISS index for the top-K most relevant documents.
  4. Fetch the matching documents from the knowledge base.
  5. Build a context string and format the prompt.
  6. Generate an answer with the Gemma-2B-IT + LoRA model.
  7. Return a structured result dict with the answer and timing metadata.

This service is the single point of entry for answering questions; all
individual services are injected via the constructor (Dependency Injection).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from app.core.config.settings import Settings
from app.core.utils.html_escape import validate_question
from app.services.embedding_service import EmbeddingService
from app.services.gemma_service import GemmaService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.logger_service import get_logger
from app.services.prompt_builder_service import PromptBuilderService
from app.services.retrieval_service import RetrievalService

logger = get_logger()


class InferenceService:
    """
    Coordinates the end-to-end Medical QA RAG pipeline.

    All ML services are injected at construction time.  This enables easy
    mocking in unit / integration tests without touching the real models.

    Attributes:
        _embedding_svc:  Produces query embeddings.
        _retrieval_svc:  Searches the FAISS index.
        _kb_svc:         Fetches documents from the knowledge base.
        _prompt_svc:     Builds context strings and formatted prompts.
        _gemma_svc:      Runs text generation.
        _settings:       Application configuration.
    """

    def __init__(
        self,
        embedding_svc: EmbeddingService,
        retrieval_svc: RetrievalService,
        kb_svc: KnowledgeBaseService,
        prompt_svc: PromptBuilderService,
        gemma_svc: GemmaService,
        settings: Settings,
    ) -> None:
        """
        Initialise the InferenceService with all required dependencies.

        Args:
            embedding_svc: Loaded :class:`EmbeddingService`.
            retrieval_svc: Loaded :class:`RetrievalService`.
            kb_svc:        Loaded :class:`KnowledgeBaseService`.
            prompt_svc:    :class:`PromptBuilderService` (stateless).
            gemma_svc:     Loaded :class:`GemmaService`.
            settings:      Application settings.
        """
        self._embedding_svc = embedding_svc
        self._retrieval_svc = retrieval_svc
        self._kb_svc = kb_svc
        self._prompt_svc = prompt_svc
        self._gemma_svc = gemma_svc
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline and return a structured answer dict.

        Pipeline steps (with per-step timing):
          1. Validate + sanitise question.
          2. Encode question → dense embedding.
          3. FAISS search → top-K retrieval results.
          4. Fetch document rows from knowledge base.
          5. Build context string.
          6. Format prompt using :attr:`PromptBuilderService.PROMPT_TEMPLATE`.
          7. Generate answer text with Gemma + LoRA.

        Args:
            question: Raw user question string (pre-validation).

        Returns:
            A dict containing:

            * ``answer`` (str):                  Generated answer text.
            * ``context_used`` (list[dict]):      Retrieved documents with
              ``index``, ``content``, and ``score`` fields.
            * ``inference_time`` (float):         Total wall-clock seconds.
            * ``retrieval_time`` (float):         FAISS search duration.
            * ``generation_time`` (float):        Gemma generation duration.
            * ``embedding_time`` (float):         Embedding duration.
            * ``documents_retrieved`` (int):      Number of context docs.

        Raises:
            ValueError: If the question fails validation (empty / too long /
                        post-sanitization empty).
            RuntimeError: If any downstream ML service fails.
        """
        t_total_start = time.perf_counter()

        # Step 1 – Validate and sanitise
        validated_q = validate_question(
            question, max_length=self._settings.MAX_QUESTION_LENGTH
        )

        # Step 2 – Embed
        t_embed_start = time.perf_counter()
        query_embedding = self._embedding_svc.encode(validated_q)
        embedding_time = time.perf_counter() - t_embed_start
        logger.log_request(
            f"Question embedded in {embedding_time:.3f}s",
            request=True,
        )

        # Step 3 – FAISS retrieval
        t_retrieve_start = time.perf_counter()
        retrieval_results = self._retrieval_svc.search(
            query_embedding, top_k=self._settings.TOP_K_RETRIEVAL
        )
        retrieval_time = time.perf_counter() - t_retrieve_start
        logger.log_request(
            f"Retrieved {len(retrieval_results)} documents in {retrieval_time:.3f}s",
            request=True,
        )

        # Step 4 – Fetch knowledge-base documents
        indices = [r["index"] for r in retrieval_results]
        documents = self._kb_svc.get_documents_by_indices(indices)

        # Step 5 – Build context
        context = self._prompt_svc.build_context(documents)

        # Step 6 – Build prompt
        prompt = self._prompt_svc.build_prompt(context, validated_q)

        # Step 7 – Generate
        t_gen_start = time.perf_counter()
        answer_text, generation_time = self._gemma_svc.generate(prompt)
        logger.log_request(
            f"Answer generated in {generation_time:.3f}s",
            request=True,
        )

        # Total time
        inference_time = time.perf_counter() - t_total_start

        # Build context_used for the response
        context_used = self._build_context_used(retrieval_results, documents)

        logger.log_request(
            f"Inference complete | "
            f"question_len={len(validated_q)} | "
            f"docs={len(documents)} | "
            f"total={inference_time:.3f}s | "
            f"embed={embedding_time:.3f}s | "
            f"retrieve={retrieval_time:.3f}s | "
            f"generate={generation_time:.3f}s",
            request=True,
        )

        return {
            "answer": answer_text,
            "context_used": context_used,
            "inference_time": round(inference_time, 4),
            "retrieval_time": round(retrieval_time, 4),
            "generation_time": round(generation_time, 4),
            "embedding_time": round(embedding_time, 4),
            "documents_retrieved": len(documents),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context_used(
        self,
        retrieval_results: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Pair retrieval results with their corresponding document content.

        Args:
            retrieval_results: List of FAISS result dicts (index, distance, score).
            documents:         List of knowledge-base row dicts.

        Returns:
            List of dicts with ``index``, ``content``, and ``score`` keys,
            suitable for serialisation in the API response.
        """
        context_used: List[Dict[str, Any]] = []
        for result, doc in zip(retrieval_results, documents):
            content = self._prompt_svc.build_context([doc])
            context_used.append(
                {
                    "index": result["index"],
                    "content": content,
                    "score": round(result["score"], 4),
                }
            )
        return context_used
