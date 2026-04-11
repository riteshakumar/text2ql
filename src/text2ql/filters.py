"""Shared filter detection utilities for all query engines.

Compiled regex patterns and stateless helper functions that are used by both
the GraphQL and SQL engines to detect filter expressions in natural language.
Centralised here to avoid copy-paste drift between engines.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Primitive pattern fragments (used to build larger expressions)
# ---------------------------------------------------------------------------

#: Matches a SQL/GraphQL field identifier (e.g. ``createdAt``, ``user_id``).
WORD_IDENTIFIER: str = r"([A-Za-z_]\w*)"

#: Matches simple scalar filter values (word chars, dots, colons, hyphens).
FILTER_VALUE: str = r"([\w.:-]+)"

#: ISO 8601 date fragment (YYYY-MM-DD).
ISO_DATE: str = r"\d{4}-\d{2}-\d{2}"

#: Words that are never valid filter values when matched in isolation.
SPURIOUS_FILTER_VALUES: frozenset[str] = frozenset(
    {"where", "with", "and", "or", "for", "of", "in", "is"}
)

# ---------------------------------------------------------------------------
# Compiled patterns – module-level singletons for performance
# ---------------------------------------------------------------------------

#: ``field >= value``  →  ``field_gte``
_RE_GTE = re.compile(rf"\b{WORD_IDENTIFIER}\s*>=\s*{FILTER_VALUE}\b")
#: ``field <= value``  →  ``field_lte``
_RE_LTE = re.compile(rf"\b{WORD_IDENTIFIER}\s*<=\s*{FILTER_VALUE}\b")
#: ``field > value``   →  ``field_gt``
_RE_GT = re.compile(rf"\b{WORD_IDENTIFIER}\s*>\s*{FILTER_VALUE}\b")
#: ``field < value``   →  ``field_lt``
_RE_LT = re.compile(rf"\b{WORD_IDENTIFIER}\s*<\s*{FILTER_VALUE}\b")

#: Lexical comparison forms (``field greater than value``, etc.)
_RE_GREATER_THAN = re.compile(rf"\b{WORD_IDENTIFIER}\s+greater than\s+{FILTER_VALUE}\b")
_RE_LESS_THAN = re.compile(rf"\b{WORD_IDENTIFIER}\s+less than\s+{FILTER_VALUE}\b")
_RE_AFTER = re.compile(rf"\b{WORD_IDENTIFIER}\s+after\s+{FILTER_VALUE}\b")
_RE_BEFORE = re.compile(rf"\b{WORD_IDENTIFIER}\s+before\s+{FILTER_VALUE}\b")

#: ``field != value`` / ``field is not value`` / ``field not value``  →  ``field_ne``
_RE_NEGATION = re.compile(
    rf"\b{WORD_IDENTIFIER}\s*(?:!=|is not|not)\s*{FILTER_VALUE}\b"
)

#: ``field between N and M``  →  ``field_gte``, ``field_lte``
_RE_BETWEEN = re.compile(
    rf"\b([A-Za-z_]+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)"
)

#: ``field between YYYY-MM-DD and YYYY-MM-DD``  →  ``field_gte``, ``field_lte``
_RE_BETWEEN_DATES = re.compile(
    rf"\b([A-Za-z_][A-Za-z0-9_]*)\s+between\s+({ISO_DATE})\s+and\s+({ISO_DATE})\b"
)

#: ``field in value1, value2``  →  ``field_in``
_RE_IN = re.compile(r"\b([a-zA-Z_]+)\s+in\s+([a-zA-Z0-9_,\s]+)")

#: ``field from YYYY-MM-DD to YYYY-MM-DD``  →  ``field_gte``, ``field_lte``
_RE_DATE_RANGE = re.compile(
    rf"\b{WORD_IDENTIFIER}\s+from\s+({ISO_DATE})\s+to\s+({ISO_DATE})\b"
)

#: Values tokenizer used by ``_RE_IN`` matches (splits on comma / "or" / "and").
_RE_IN_SPLIT = re.compile(r",|\s+or\s+|\s+and\s+")

# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------


def detect_comparison_filters(lowered: str) -> dict[str, Any]:
    """Return comparison filters (``_gte``, ``_lte``, ``_gt``, ``_lt``) found in *lowered*.

    Handles both symbolic (``>=``) and lexical (``greater than``) forms.
    """
    filters: dict[str, Any] = {}
    symbolic = [
        (_RE_GTE, "_gte"),
        (_RE_LTE, "_lte"),
        (_RE_GT, "_gt"),
        (_RE_LT, "_lt"),
    ]
    for pattern, suffix in symbolic:
        for m in pattern.finditer(lowered):
            filters[f"{m.group(1)}{suffix}"] = m.group(2)

    lexical = [
        (_RE_GREATER_THAN, "_gt"),
        (_RE_LESS_THAN, "_lt"),
        (_RE_AFTER, "_gt"),
        (_RE_BEFORE, "_lt"),
    ]
    for pattern, suffix in lexical:
        for m in pattern.finditer(lowered):
            filters[f"{m.group(1)}{suffix}"] = m.group(2)

    return filters


def detect_negation_filters(lowered: str) -> dict[str, Any]:
    """Return inequality filters (``_ne``) found in *lowered*."""
    filters: dict[str, Any] = {}
    for m in _RE_NEGATION.finditer(lowered):
        filters[f"{m.group(1)}_ne"] = m.group(2)
    return filters


def detect_between_filters(lowered: str) -> dict[str, Any]:
    """Return range filters from ``BETWEEN … AND …`` expressions in *lowered*.

    Handles both numeric and ISO-date operands.
    """
    filters: dict[str, Any] = {}
    for m in _RE_BETWEEN_DATES.finditer(lowered):
        field = m.group(1)
        filters[f"{field}_gte"] = m.group(2)
        filters[f"{field}_lte"] = m.group(3)
    for m in _RE_BETWEEN.finditer(lowered):
        field = m.group(1)
        # Don't double-write if the date pattern already captured this field.
        if f"{field}_gte" not in filters:
            filters[f"{field}_gte"] = m.group(2)
            filters[f"{field}_lte"] = m.group(3)
    return filters


def detect_in_filters(lowered: str) -> dict[str, Any]:
    """Return set-membership filters (``_in``) found in *lowered*."""
    filters: dict[str, Any] = {}
    for m in _RE_IN.finditer(lowered):
        field = m.group(1)
        values = [t.strip() for t in _RE_IN_SPLIT.split(m.group(2)) if t.strip()]
        if values:
            filters[f"{field}_in"] = values
    return filters


def detect_date_range_filters(lowered: str) -> dict[str, Any]:
    """Return date-range filters from ``field from DATE to DATE`` patterns."""
    filters: dict[str, Any] = {}
    for m in _RE_DATE_RANGE.finditer(lowered):
        field = m.group(1)
        filters[f"{field}_gte"] = m.group(2)
        filters[f"{field}_lte"] = m.group(3)
    return filters
