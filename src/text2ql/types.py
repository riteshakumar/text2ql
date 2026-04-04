from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryRequest:
    """Normalized text-to-query input."""

    text: str
    target: str = "graphql"
    schema: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResult:
    """Generated query plus metadata."""

    query: str
    target: str
    confidence: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)
