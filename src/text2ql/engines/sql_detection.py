from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text2ql.engines.sql import SQLEngine


def detect_table(engine: "SQLEngine", lowered: str, config: Any) -> str:
    for alias, canonical in engine._sorted_alias_pairs(config.entity_aliases):
        if engine._contains_token(lowered, alias):
            return canonical
    for entity in config.entities:
        if engine._contains_entity_token(lowered, entity.lower()):
            return entity
    inferred_from_values = _infer_table_from_filter_value_aliases(engine, lowered, config)
    if inferred_from_values:
        return inferred_from_values
    inferred = _infer_table_from_column_mentions(engine, lowered, config)
    if inferred:
        return inferred
    if config.default_entity:
        return config.default_entity
    return config.entities[0] if config.entities else engine._extract_entity_from_text(lowered)


def _infer_table_from_column_mentions(engine: "SQLEngine", lowered: str, config: Any) -> str | None:
    """Infer table from explicit column mentions when entity is not named."""
    candidates: list[tuple[str, int, int]] = []
    field_alias_pairs = engine._sorted_alias_pairs(config.field_aliases)

    for entity in config.entities:
        columns = engine._columns_for_table(config, entity)
        if not columns:
            continue

        score = 0
        for column in columns:
            if engine._contains_column_reference(lowered, column):
                score += 2

        for alias, canonical in field_alias_pairs:
            if canonical in columns and engine._contains_column_reference(lowered, alias):
                score += 1

        if score > 0:
            candidates.append((entity, score, len(columns)))

    if not candidates:
        return None

    max_score = max(score for _, score, _ in candidates)
    top = [(entity, width) for entity, score, width in candidates if score == max_score]
    narrowest_width = min(width for _, width in top)
    narrowest = [entity for entity, width in top if width == narrowest_width]
    if len(narrowest) == 1:
        return narrowest[0]
    return None


def _infer_table_from_filter_value_aliases(engine: "SQLEngine", lowered: str, config: Any) -> str | None:
    value_aliases = getattr(config, "filter_value_aliases", {})
    if not isinstance(value_aliases, dict) or not value_aliases:
        return None

    scores: dict[str, float] = {}
    widths: dict[str, int] = {}
    for canonical, alias_map in value_aliases.items():
        if not isinstance(alias_map, dict):
            continue
        if not any(_matches_value_alias(lowered, str(alias)) for alias in alias_map.keys()):
            continue
        for entity in getattr(config, "entities", []):
            if not _entity_supports_filter_key(engine, config, entity, str(canonical)):
                continue
            scores[entity] = scores.get(entity, 0.0) + 1.0
            widths[entity] = min(widths.get(entity, 10_000), len(engine._columns_for_table(config, entity)))

    if not scores:
        return None

    max_score = max(scores.values())
    top = [entity for entity, score in scores.items() if score == max_score]
    if len(top) == 1:
        return top[0]

    narrowest_width = min(widths.get(entity, 10_000) for entity in top)
    narrowed = [entity for entity in top if widths.get(entity, 10_000) == narrowest_width]
    if len(narrowed) == 1:
        return narrowed[0]
    return None


def detect_columns(engine: "SQLEngine", lowered: str, config: Any, table: str) -> list[str]:
    allowed = engine._columns_for_table(config, table)
    selected: list[str] = []
    for column in allowed:
        if engine._contains_column_reference(lowered, column):
            selected.append(column)
    for alias, canonical in engine._sorted_alias_pairs(config.field_aliases):
        if canonical in allowed and engine._contains_column_reference(lowered, alias):
            selected.append(canonical)
    selected = engine._unique_in_order(selected)
    if selected:
        cash_projection = _select_cash_holdings_projection(
            engine=engine,
            lowered=lowered,
            table=table,
            allowed=allowed,
            selected=selected,
        )
        if cash_projection:
            return cash_projection
        return selected
    by_entity_defaults = getattr(config, "default_fields_by_entity", {}).get(table, [])
    if by_entity_defaults:
        defaults = [field for field in by_entity_defaults if field in allowed]
        if defaults:
            return defaults
    if config.default_fields:
        defaults = [field for field in config.default_fields if field in allowed]
        if defaults:
            return defaults
    recent_snapshot = _select_recent_snapshot_columns(lowered, allowed)
    if recent_snapshot:
        return recent_snapshot
    semantic = _select_semantic_columns(lowered, allowed, table=table)
    if semantic:
        cash_projection = _select_cash_holdings_projection(
            engine=engine,
            lowered=lowered,
            table=table,
            allowed=allowed,
            selected=semantic,
        )
        if cash_projection:
            return cash_projection
        return semantic
    return _ranked_fallback_columns(allowed)


def detect_order(
    engine: "SQLEngine",
    lowered: str,
    selected_columns: list[str],
    all_columns: list[str] | None = None,
) -> tuple[str | None, str | None]:
    if "latest order" in lowered:
        return None, None
    if any(token in lowered for token in ("latest", "newest", "most recent")):
        return engine._detect_order_field(lowered, selected_columns, all_columns), "DESC"
    highest = re.search(r"\bhighest\s+([A-Za-z_]\w*)\b", lowered)
    if highest:
        return highest.group(1), "DESC"
    lowest = re.search(r"\blowest\s+([A-Za-z_]\w*)\b", lowered)
    if lowest:
        return lowest.group(1), "ASC"
    return None, None


def _select_semantic_columns(
    lowered: str,
    allowed: list[str],
    *,
    table: str = "",
    max_fields: int = 3,
) -> list[str]:
    if not allowed:
        return []
    prompt_tokens = _expanded_tokens(_tokenize(lowered))
    # Avoid selecting fields solely because prompt repeats the entity name.
    if table:
        prompt_tokens -= _expanded_tokens(_tokenize(table))
    if not prompt_tokens:
        return []
    scored: list[tuple[float, int, str]] = []
    for idx, column in enumerate(allowed):
        score = _column_semantic_score(prompt_tokens, column)
        if score <= 0:
            continue
        scored.append((score, idx, column))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, _, column in scored[:max_fields]]


def _select_cash_holdings_projection(
    engine: "SQLEngine",
    lowered: str,
    table: str,
    allowed: list[str],
    selected: list[str],
) -> list[str]:
    """Prefer useful holdings snapshots for prompts like 'cash positions only'.

    Without this guard, lexical matching often selects only ``isCash``, which is
    valid but less useful than returning the matching positions themselves.
    """
    if len(selected) != 1 or str(selected[0]).lower() != "iscash":
        return []
    if "cash" not in lowered:
        return []
    if "only" not in lowered and "position" not in lowered:
        return []
    if _is_explicit_is_cash_request(lowered):
        return []
    if engine._score_holdings_table(table, allowed) <= 0:
        return []

    lower_to_original = {str(column).lower(): str(column) for column in allowed}
    preferred = (
        "quantity",
        "symbol",
        "securityDescription",
        "shortDesc",
        "mobileDesc",
        "desc",
    )
    projected: list[str] = []
    for candidate in preferred:
        match = lower_to_original.get(candidate.lower())
        if match is not None and match not in projected:
            projected.append(match)
        if len(projected) >= 2:
            break
    return projected


def _is_explicit_is_cash_request(lowered: str) -> bool:
    return "iscash" in lowered or re.search(r"\bis\s+cash\b", lowered) is not None


def _select_recent_snapshot_columns(lowered: str, allowed: list[str]) -> list[str]:
    if not any(token in lowered for token in ("latest", "newest", "most recent")):
        return []
    lower_to_original = {str(column).lower(): str(column) for column in allowed}
    priority = (
        "quantity",
        "symbol",
        "securityDescription",
        "shortDesc",
        "mobileDesc",
        "desc",
        "net",
        "amount",
        "price",
        "principal",
    )
    selected: list[str] = []
    for candidate in priority:
        match = lower_to_original.get(candidate.lower())
        if match is not None and match not in selected:
            selected.append(match)
        if len(selected) >= 2:
            break
    return selected


def _column_semantic_score(prompt_tokens: set[str], column: str) -> float:
    column_tokens = _expanded_tokens(_tokenize(column))
    if not column_tokens:
        return 0.0
    overlap = len(prompt_tokens.intersection(column_tokens))
    if overlap == 0:
        return 0.0
    score = overlap / max(1, len(column_tokens))
    if _is_detail_container(column):
        score -= 0.35
    if _looks_scalar_column(column):
        score += 0.05
    return score


def _ranked_fallback_columns(allowed: list[str], max_fields: int = 3) -> list[str]:
    if not allowed:
        return ["id"]
    scored: list[tuple[float, int, str]] = []
    for idx, column in enumerate(allowed):
        score = _fallback_column_score(column)
        scored.append((score, idx, column))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, _, column in scored[:max_fields]]


def _fallback_column_score(column: str) -> float:
    lowered = str(column).lower()
    score = 0.0
    if _is_detail_container(lowered):
        score -= 2.0
    if any(token in lowered for token in ("date", "time", "timestamp", "created", "updated", "posted", "traded")):
        score += 0.9
    if any(token in lowered for token in ("symbol", "quantity", "amount", "price", "value", "balance", "status")):
        score += 0.8
    if lowered in {"id", "name", "title"}:
        score += 0.7
    if lowered.endswith(("id", "num", "code")):
        score += 0.3
    return score


def _is_detail_container(column: str) -> bool:
    lowered = str(column).lower()
    return lowered.endswith("detail")


def _looks_scalar_column(column: str) -> bool:
    lowered = str(column).lower()
    return not lowered.endswith("detail")


def _tokenize(text: str) -> set[str]:
    with_spaces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    with_spaces = with_spaces.replace("_", " ")
    return {token for token in re.findall(r"[a-z0-9]+", with_spaces.lower()) if token}


def _expanded_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    synonyms = {
        "txn": "transaction",
        "transaction": "txn",
        "acct": "account",
        "account": "acct",
        "amt": "amount",
        "amount": "amt",
        "qty": "quantity",
        "quantity": "qty",
        "desc": "description",
        "description": "desc",
        "bal": "balance",
        "balance": "bal",
        "mkt": "market",
        "market": "mkt",
        "chg": "change",
        "change": "chg",
        "avail": "available",
        "available": "avail",
        "num": "number",
        "number": "num",
    }
    for token in tokens:
        mapped = synonyms.get(token)
        if mapped:
            expanded.add(mapped)
    return expanded


def _entity_supports_filter_key(engine: "SQLEngine", config: Any, entity: str, candidate_key: str) -> bool:
    canonical = str(candidate_key).lower()
    columns = [str(col).lower() for col in engine._columns_for_table(config, entity)]
    # SQL table inference should only consider physically queryable columns.
    # Args-only matches can route to entities that are not materialized as tables.
    return canonical in columns


def _matches_value_alias(lowered: str, alias: str) -> bool:
    alias_text = str(alias).strip().lower()
    if not alias_text:
        return False
    if re.search(rf"\b{re.escape(alias_text)}\b", lowered):
        return True
    alias_tokens = [token for token in re.findall(r"[a-z0-9]+", alias_text) if len(token) >= 4]
    if not alias_tokens:
        return False
    lowered_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    if len(alias_tokens) <= 3:
        return all(token in lowered_tokens for token in alias_tokens)
    overlap = sum(1 for token in alias_tokens if token in lowered_tokens)
    return overlap >= 2
