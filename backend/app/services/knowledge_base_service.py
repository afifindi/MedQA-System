"""
Medical QA System - Knowledge Base Service.

Loads the medical knowledge base from a CSV file and provides a clean
interface for fetching document rows by their integer index, which is the
format returned by the :class:`RetrievalService` after a FAISS search.

The CSV is loaded once at startup and held in memory as a Pandas DataFrame.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.core.config.settings import Settings
from app.services.logger_service import get_logger

logger = get_logger()


class KnowledgeBaseService:
    """
    In-memory knowledge-base store backed by a Pandas DataFrame.

    Reads ``knowledge_base.csv`` from the pre-trained artefact directory and
    exposes methods to retrieve rows by index, count documents, and inspect
    column names.

    Attributes:
        _settings:  Application configuration.
        _df:        The loaded DataFrame (None until :meth:`load` is called).
        _is_loaded: Whether the DataFrame has been successfully loaded.
        _load_time: Seconds taken to load the CSV.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialise the KnowledgeBaseService without loading data.

        Args:
            settings: The application settings object.
        """
        self._settings = settings
        self._df: Optional[pd.DataFrame] = None
        self._is_loaded: bool = False
        self._load_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the knowledge base CSV into a Pandas DataFrame.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            RuntimeError:      On any other I/O or parsing error.
        """
        t_start = time.perf_counter()
        kb_path = Path(self._settings.KNOWLEDGE_BASE_PATH)

        logger.log_model(f"Loading knowledge base from: {kb_path.resolve()}")

        try:
            self._df = pd.read_csv(str(kb_path))
            self._load_time = time.perf_counter() - t_start
            self._is_loaded = True
            logger.log_model(
                f"Knowledge base loaded.  "
                f"Rows: {len(self._df)}, "
                f"Columns: {self._df.columns.tolist()}, "
                f"Load time: {self._load_time:.2f}s"
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load knowledge base CSV from '{kb_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Data Access
    # ------------------------------------------------------------------

    def get_documents_by_indices(self, indices: List[int]) -> List[Dict]:
        """
        Retrieve knowledge-base rows by their zero-based integer indices.

        Out-of-range indices are silently skipped to prevent crashes when a
        FAISS index references stale rows.

        Args:
            indices: List of zero-based row indices (from FAISS search).

        Returns:
            List of dicts, each representing one CSV row.  The keys are the
            CSV column names and the values are the corresponding cell values.

        Raises:
            RuntimeError: If the knowledge base has not been loaded.
        """
        self._assert_loaded()
        results: List[Dict] = []
        for idx in indices:
            if 0 <= idx < len(self._df):  # type: ignore[arg-type]
                results.append(self._df.iloc[idx].to_dict())  # type: ignore[union-attr]
            else:
                logger.log_error(
                    f"FAISS returned out-of-range index {idx} "
                    f"(knowledge base has {len(self._df)} rows)."  # type: ignore[arg-type]
                )
        return results

    def get_document_count(self) -> int:
        """
        Return the total number of documents in the knowledge base.

        Returns:
            Integer count, or 0 if the KB has not been loaded.
        """
        if self._df is None:
            return 0
        return len(self._df)

    def get_columns(self) -> List[str]:
        """
        Return the column names of the knowledge base CSV.

        Returns:
            List of column name strings, or an empty list if not loaded.
        """
        if self._df is None:
            return []
        return self._df.columns.tolist()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the knowledge base has been successfully loaded."""
        return self._is_loaded

    @property
    def load_time(self) -> float:
        """Seconds elapsed during knowledge base loading."""
        return self._load_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if the knowledge base has not been loaded."""
        if not self._is_loaded or self._df is None:
            raise RuntimeError(
                "KnowledgeBaseService.load() must be called before accessing data."
            )
