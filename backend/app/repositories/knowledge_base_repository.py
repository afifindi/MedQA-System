"""
Medical QA System - Knowledge Base Repository.

Provides a clean data-access interface on top of :class:`KnowledgeBaseService`.
Decouples the API/controller layer from the underlying Pandas DataFrame,
making it straightforward to swap the storage back-end in the future.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.services.knowledge_base_service import KnowledgeBaseService


class KnowledgeBaseRepository:
    """
    Repository wrapping :class:`KnowledgeBaseService` for document access.

    Follows the Repository pattern from Domain-Driven Design: callers interact
    with this object rather than directly with the service, keeping the API
    layer agnostic of the storage implementation.

    Attributes:
        _kb_svc: The underlying knowledge-base service.
    """

    def __init__(self, kb_svc: KnowledgeBaseService) -> None:
        """
        Initialise the repository with an injected KnowledgeBaseService.

        Args:
            kb_svc: A loaded :class:`KnowledgeBaseService` instance.
        """
        self._kb_svc = kb_svc

    def find_by_indices(self, indices: List[int]) -> List[Dict]:
        """
        Retrieve documents by their zero-based integer indices.

        Delegates directly to :meth:`KnowledgeBaseService.get_documents_by_indices`.

        Args:
            indices: List of row indices returned by the FAISS search.

        Returns:
            List of document dicts.
        """
        return self._kb_svc.get_documents_by_indices(indices)

    def count(self) -> int:
        """
        Return the total number of documents in the knowledge base.

        Returns:
            Integer document count.
        """
        return self._kb_svc.get_document_count()

    def get_columns(self) -> List[str]:
        """
        Return the column names of the underlying knowledge base CSV.

        Returns:
            List of column name strings.
        """
        return self._kb_svc.get_columns()

    def search_by_keyword(
        self, keyword: str, column: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform a case-insensitive keyword search over the knowledge base.

        If *column* is specified, only that column is searched.  Otherwise,
        all string-typed columns are searched and matching rows are returned.

        Args:
            keyword: The search term.
            column:  Optional column name to restrict the search to.

        Returns:
            List of matching document dicts.  Empty list if nothing matches
            or if the knowledge base has not been loaded.
        """
        # Access internal DataFrame via the service (justified by tight coupling)
        df = self._kb_svc._df  # noqa: SLF001
        if df is None:
            return []

        if column and column in df.columns:
            mask = df[column].astype(str).str.contains(
                keyword, case=False, na=False, regex=False
            )
        else:
            # Search all object (string) columns
            mask = df.select_dtypes(include="object").apply(
                lambda col: col.astype(str).str.contains(
                    keyword, case=False, na=False, regex=False
                )
            ).any(axis=1)

        return df[mask].to_dict(orient="records")
