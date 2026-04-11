from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryRequest:
    """Normalized text-to-query input."""

    text: str
    target: str = "graphql"
    schema: dict[str, Any] | None = None
    mapping: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResult:
    """Generated query plus metadata."""

    query: str
    target: str
    confidence: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ValidationError(ValueError):
    """Raised when query components fail schema validation in strict mode.

    This is raised by :class:`~text2ql.engines.sql.SQLEngine` and
    :class:`~text2ql.engines.graphql.GraphQLEngine` when ``strict_validation``
    is ``True`` and one or more of the following conditions are detected:

    - Contradictory filter values for the same field (e.g.
      ``status = 'active' AND status = 'inactive'``).
    - JOIN ON-clause columns that do not exist in the referenced tables.
    - Any other schema-level inconsistency that would produce an invalid query.

    In non-strict mode (the default) these conditions are recorded in
    ``QueryResult.metadata["validation_notes"]`` and the engine degrades
    gracefully rather than raising.

    Parameters
    ----------
    message:
        Human-readable summary of all issues found.
    issues:
        Individual validation issue strings (one per detected problem).
    """

    def __init__(self, message: str, issues: list[str]) -> None:
        super().__init__(message)
        self.issues: list[str] = issues

    def __str__(self) -> str:  # pragma: no cover
        lines = [super().__str__()]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        return "\n".join(lines)
