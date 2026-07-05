"""
Medical QA System - FAISS Retrieval Service.

Loads the pre-built FAISS index and the knowledge-base embedding matrix to
perform fast approximate nearest-neighbour searches.  Given a query embedding
produced by the :class:`EmbeddingService`, this service returns the indices
and similarity scores of the top-K most relevant documents.

The FAISS index and embeddings are never rebuilt by the application; they are
loaded exactly as shipped in ``models/gemma_medical_qa_final/``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.core.config.settings import Settings
from app.services.logger_service import get_logger

logger = get_logger()


class RetrievalService:
    """
    FAISS-based semantic retrieval service.

    Loads a pre-built FAISS index and the associated knowledge-base embedding
    matrix at startup.  Provides a :meth:`search` method that accepts a query
    embedding and returns the top-K nearest-neighbour results with L2-derived
    similarity scores.

    Attributes:
        _settings:    Application configuration.
        _index:       The loaded FAISS index object.
        _embeddings:  Pre-computed knowledge-base embeddings matrix.
        _is_loaded:   Whether the index has been successfully loaded.
        _load_time:   Seconds taken to load the index.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialise the RetrievalService without loading the FAISS index.

        Args:
            settings: The application settings object.
        """
        self._settings = settings
        self._index: Optional[object] = None
        self._embeddings: Optional[np.ndarray] = None
        self._is_loaded: bool = False
        self._load_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the FAISS index and knowledge-base embeddings from disk.

        Reads ``settings.FAISS_PATH`` (``faiss_index.bin``) and
        ``settings.KB_EMBEDDINGS_PATH`` (``kb_embeddings.npy``).  Both files
        must exist; they are validated by :class:`ConfigurationService` before
        this method is called.

        Raises:
            FileNotFoundError: If either file is missing at load time.
            RuntimeError:      On any other I/O or FAISS error.
        """
        import faiss  # noqa: F401  (imported here to isolate the dependency)

        t_start = time.perf_counter()

        faiss_path = Path(self._settings.FAISS_PATH)
        emb_path = Path(self._settings.KB_EMBEDDINGS_PATH)

        try:
            logger.log_model(f"Loading FAISS index from: {faiss_path.resolve()}")
            self._index = faiss.read_index(str(faiss_path))
            logger.log_model(
                f"FAISS index loaded.  "
                f"Total vectors: {self._index.ntotal}, "  # type: ignore[union-attr]
                f"Dimension: {self._index.d}"  # type: ignore[union-attr]
            )

            logger.log_model(f"Loading KB embeddings from: {emb_path.resolve()}")
            self._embeddings = np.load(str(emb_path)).astype(np.float32)
            logger.log_model(
                f"KB embeddings loaded.  Shape: {self._embeddings.shape}"
            )

            self._load_time = time.perf_counter() - t_start
            self._is_loaded = True
            logger.log_model(
                f"RetrievalService ready.  Load time: {self._load_time:.2f}s"
            )

        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAISS index or embeddings: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[dict]:
        """
        Search the FAISS index for the *top_k* nearest neighbours.

        Args:
            query_embedding: A 1-D float32 array of shape ``(embedding_dim,)``
                             produced by :meth:`EmbeddingService.encode`.
            top_k:           Number of results to return (default 3).

        Returns:
            A list of dicts, each containing:

            * ``index`` (int):    Row index into the knowledge base CSV.
            * ``distance`` (float): Raw L2 distance (lower = more similar).
            * ``score`` (float): Similarity score computed as
              ``1 / (1 + distance)``; in the range ``(0, 1]``.

            Results are sorted by ascending distance (most similar first).
            Invalid FAISS indices (``-1``) are filtered out.

        Raises:
            RuntimeError: If the index has not been loaded.
        """
        self._assert_loaded()

        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query, top_k)  # type: ignore[union-attr]

        results: List[dict] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                # FAISS returns -1 for padding when there are fewer vectors
                # than top_k
                continue
            results.append(
                {
                    "index": int(idx),
                    "distance": float(dist),
                    "score": float(1.0 / (1.0 + dist)),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the FAISS index has been successfully loaded."""
        return self._is_loaded

    @property
    def load_time(self) -> float:
        """Seconds elapsed during index loading."""
        return self._load_time

    @property
    def index_size(self) -> int:
        """Number of vectors in the FAISS index (0 if not loaded)."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if the index has not been loaded."""
        if not self._is_loaded or self._index is None:
            raise RuntimeError(
                "RetrievalService.load() must be called before searching."
            )
