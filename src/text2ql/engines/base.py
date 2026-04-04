from __future__ import annotations

from abc import ABC, abstractmethod

from text2ql.types import QueryRequest, QueryResult


class QueryEngine(ABC):
    """Converts a normalized request into a target query language."""

    @abstractmethod
    def generate(self, request: QueryRequest) -> QueryResult:
        """Generate a query."""
