from __future__ import annotations

from typing import Callable


def quantity_candidates() -> tuple[str, ...]:
    return ("quantity", "qty", "shares", "units", "amount", "holding")


def identifier_candidates() -> tuple[str, ...]:
    return ("symbol", "ticker", "stock", "asset", "security", "code", "name")


def score_holdings_container(
    container_name: str,
    fields: list[str],
    *,
    quantity_keys: tuple[str, ...] | None = None,
    identifier_keys: tuple[str, ...] | None = None,
) -> int:
    quantity_keys = quantity_keys or quantity_candidates()
    identifier_keys = identifier_keys or identifier_candidates()
    lowered_name = container_name.lower()
    lowered_fields = {field.lower() for field in fields}
    has_identifier = any(candidate in lowered_fields for candidate in identifier_keys)
    has_quantity = any(candidate in lowered_fields for candidate in quantity_keys)
    if not (has_identifier and has_quantity):
        return 0
    score = 1
    if lowered_name in {"positions", "holdings", "assets"}:
        score += 4
    if {"symbol", "securitytype"}.intersection(lowered_fields):
        score += 1
    if {"acctnum", "acctname", "accountpositioncount"}.intersection(lowered_fields):
        score -= 3
    return score


def resolve_holdings_container(
    container_names: list[str],
    fields_for_container: Callable[[str], list[str]],
    *,
    quantity_keys: tuple[str, ...] | None = None,
    identifier_keys: tuple[str, ...] | None = None,
) -> str | None:
    best_name: str | None = None
    best_score = 0
    for name in container_names:
        fields = fields_for_container(name)
        if not fields:
            continue
        score = score_holdings_container(
            name,
            fields,
            quantity_keys=quantity_keys,
            identifier_keys=identifier_keys,
        )
        if score > best_score:
            best_score = score
            best_name = name
    return best_name if best_score > 0 else None


def resolve_identifier_filter_key(
    *,
    args: list[str],
    fields: list[str],
    candidate_aliases: dict[str, str] | None = None,
    identifier_keys: tuple[str, ...] | None = None,
) -> str | None:
    candidate_aliases = candidate_aliases or {}
    identifier_keys = identifier_keys or identifier_candidates()
    lowered_args = {arg.lower(): arg for arg in args}
    lowered_fields = {field.lower(): field for field in fields}
    for candidate in identifier_keys:
        canonical = candidate_aliases.get(candidate)
        if not (isinstance(canonical, str) and canonical):
            continue
        canonical_lower = canonical.lower()
        if canonical_lower in lowered_args:
            return lowered_args[canonical_lower]
        if canonical_lower in lowered_fields:
            return lowered_fields[canonical_lower]
    for candidate in identifier_keys:
        if candidate in lowered_args:
            return lowered_args[candidate]
        if candidate in lowered_fields:
            return lowered_fields[candidate]
    return None


def resolve_holdings_projection(
    fields: list[str],
    *,
    quantity_keys: tuple[str, ...] | None = None,
    identifier_keys: tuple[str, ...] | None = None,
) -> list[str]:
    quantity_keys = quantity_keys or quantity_candidates()
    identifier_keys = identifier_keys or identifier_candidates()
    lowered_to_original = {field.lower(): field for field in fields}
    selection: list[str] = []
    for candidate in quantity_keys:
        field = lowered_to_original.get(candidate)
        if field is not None:
            selection.append(field)
            break
    for candidate in identifier_keys:
        field = lowered_to_original.get(candidate)
        if field is not None and field not in selection:
            selection.append(field)
            break
    return selection
