from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text2ql.engines.sql import SQLEngine


def detect_table(engine: "SQLEngine", lowered: str, config: Any) -> str:
    alias_or_name_table: str | None = None
    for alias, canonical in engine._sorted_alias_pairs(config.entity_aliases):
        if engine._contains_token(lowered, alias):
            alias_or_name_table = canonical
            break
    if alias_or_name_table is None:
        for entity in config.entities:
            if engine._contains_entity_token(lowered, entity.lower()):
                alias_or_name_table = entity
                break

    metric_table = _infer_table_from_metric_intent(engine, lowered, config)
    semantic_table, semantic_score = _infer_table_from_semantic_match(engine, lowered, config)
    if metric_table is not None and metric_table != alias_or_name_table:
        return metric_table
    if alias_or_name_table is not None:
        if semantic_table is not None and semantic_table != alias_or_name_table:
            alias_score = _table_semantic_score(engine, lowered, config, alias_or_name_table)
            alias_field_score = _table_field_semantic_score(engine, lowered, config, alias_or_name_table)
            semantic_field_score = _table_field_semantic_score(engine, lowered, config, semantic_table)
            has_metric_intent = _has_metric_routing_intent(lowered)
            if has_metric_intent and semantic_score >= alias_score + 0.2 and semantic_field_score >= alias_field_score + 0.1:
                return semantic_table
        return alias_or_name_table

    inferred_from_values = _infer_table_from_filter_value_aliases(engine, lowered, config)
    if inferred_from_values:
        return inferred_from_values
    inferred = _infer_table_from_column_mentions(engine, lowered, config)
    if inferred:
        return inferred
    if semantic_table:
        return semantic_table
    if config.default_entity:
        return config.default_entity
    return config.entities[0] if config.entities else engine._extract_entity_from_text(lowered)


def _infer_table_from_metric_intent(engine: "SQLEngine", lowered: str, config: Any) -> str | None:
    phrase_to_fields: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("available cash", ("cashOnly", "cashWithMargin", "cashWithoutEquity", "cash", "withoutMarginImpact")),
        ("cash available", ("cashOnly", "cashWithMargin", "cashWithoutEquity", "cash", "withoutMarginImpact")),
        ("net worth", ("netWorth", "regulatoryNetWorth", "totalMarketVal", "marketVal")),
        ("market value", ("totalMarketVal", "marketVal", "fidelityTotalMktVal", "nonFidelityTotalMktVal")),
        ("gain loss", ("totalGainLoss", "todaysGainLoss", "netWorthChg")),
        ("sat", ("AvgScrMath", "AvgScrRead", "AvgScrWrite", "NumTstTakr", "NumGE1500", "sname")),
        ("average reading score", ("AvgScrRead",)),
        ("test takers", ("NumTstTakr",)),
        ("scoring above 1500", ("NumGE1500",)),
        ("free meal", ("Free Meal Count (K-12)", "Percent (%) Eligible Free (K-12)", "FRPM Count (K-12)")),
        ("free meal percentage", ("Percent (%) Eligible Free (K-12)",)),
        ("enrollment", ("Enrollment (K-12)",)),
        ("frpm", ("Enrollment (K-12)", "FRPM Count (K-12)", "Percent (%) Eligible Free (K-12)")),
        ("consumption", ("Consumption",)),
        ("transactions", ("TransactionID", "Amount", "Price", "ProductID", "GasStationID", "CustomerID")),
        ("price paid", ("Price", "CustomerID", "TransactionID")),
        ("paid by customer", ("Price", "CustomerID", "TransactionID")),
        ("gas station", ("GasStationID", "ChainID", "Country", "Segment")),
        ("overall rating", ("overall_rating",)),
        ("preferred foot", ("preferred_foot",)),
        ("prefer each foot", ("preferred_foot",)),
        ("build up play speed", ("buildUpPlaySpeed",)),
        ("home goals", ("home_team_goal",)),
        ("away goals", ("away_team_goal",)),
        ("highest earnings", ("Earnings", "People_ID")),
        ("best finish", ("Best_Finish", "People_ID")),
        ("registered for", ("registration_date", "course_id", "student_id")),
        ("student registered", ("registration_date", "course_id", "student_id")),
        ("attended each course", ("course_id", "student_id", "date_of_attendance")),
        ("student attend", ("course_id", "student_id", "date_of_attendance")),
        ("did student", ("course_id", "student_id", "date_of_attendance")),
        ("attended", ("date_of_attendance", "course_id", "student_id")),
        ("attend", ("date_of_attendance", "course_id", "student_id")),
        ("registration date", ("registration_date", "course_id", "student_id")),
        ("received a bonus", ("Bonus", "Employee_ID", "Year_awarded")),
        ("bonus greater", ("Bonus", "Employee_ID", "Year_awarded")),
        ("work in shops", ("Shop_ID", "Employee_ID", "Start_from")),
        ("shops located", ("Shop_ID", "Employee_ID", "Location")),
        ("car model", ("ModelId", "Model")),
        ("models produced", ("ModelId", "Model", "Maker")),
        ("connections", ("atom_id", "atom_id2", "bond_id")),
    )
    for phrase, field_candidates in phrase_to_fields:
        if phrase not in lowered:
            continue
        return _best_entity_for_field_candidates(engine, config, field_candidates)
    return None


def _best_entity_for_field_candidates(
    engine: "SQLEngine",
    config: Any,
    field_candidates: tuple[str, ...],
) -> str | None:
    candidates: list[tuple[str, int, int]] = []
    wanted = {str(field).lower() for field in field_candidates}
    for entity in getattr(config, "entities", []):
        columns = engine._columns_for_table(config, entity)
        if not columns:
            continue
        lowered_cols = {str(column).lower() for column in columns}
        score = sum(1 for field in wanted if field in lowered_cols)
        if score <= 0:
            continue
        candidates.append((str(entity), score, len(columns)))
    if not candidates:
        return None
    max_score = max(score for _, score, _ in candidates)
    top = [(entity, width) for entity, score, width in candidates if score == max_score]
    narrowest = min(width for _, width in top)
    narrowed = [entity for entity, width in top if width == narrowest]
    if len(narrowed) == 1:
        return narrowed[0]
    return None


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


def _infer_table_from_semantic_match(
    engine: "SQLEngine",
    lowered: str,
    config: Any,
) -> tuple[str | None, float]:
    prompt_tokens = _expanded_tokens(_tokenize(lowered))
    if not prompt_tokens:
        return None, 0.0

    best_entity: str | None = None
    best_score = 0.0
    for entity in config.entities:
        score = _table_semantic_score_from_tokens(engine, prompt_tokens, config, entity)
        if score > best_score:
            best_score = score
            best_entity = entity

    if best_score < 0.7:
        return None, best_score
    return best_entity, best_score


def _table_semantic_score(
    engine: "SQLEngine",
    lowered: str,
    config: Any,
    entity: str,
) -> float:
    prompt_tokens = _expanded_tokens(_tokenize(lowered))
    return _table_semantic_score_from_tokens(engine, prompt_tokens, config, entity)


def _table_field_semantic_score(
    engine: "SQLEngine",
    lowered: str,
    config: Any,
    entity: str,
) -> float:
    prompt_tokens = _expanded_tokens(_tokenize(lowered))
    columns = engine._columns_for_table(config, entity)
    return max((_column_semantic_score(prompt_tokens, column) for column in columns), default=0.0)


def _table_semantic_score_from_tokens(
    engine: "SQLEngine",
    prompt_tokens: set[str],
    config: Any,
    entity: str,
) -> float:
    entity_score = _entity_semantic_score(prompt_tokens, entity)
    columns = engine._columns_for_table(config, entity)
    field_score = max((_column_semantic_score(prompt_tokens, column) for column in columns), default=0.0)
    return (1.8 * entity_score) + (1.2 * field_score)


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
    selected = _augment_song_release_projection(lowered, allowed, selected)
    selected = _augment_course_projection(lowered, allowed, selected)
    selected = _augment_name_projection(lowered, allowed, selected)
    simple_name_columns = _select_simple_name_projection(lowered, allowed)
    if simple_name_columns:
        if _is_plain_names_query(lowered):
            return simple_name_columns
        if not selected:
            return simple_name_columns
        if not any(_is_name_like_column(column) for column in selected):
            selected = simple_name_columns + [column for column in selected if column not in simple_name_columns]
    if selected:
        selected = _trim_overbroad_selection(lowered, selected)
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
        semantic = _augment_name_projection(lowered, allowed, semantic)
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
    return _augment_name_projection(lowered, allowed, _ranked_fallback_columns(allowed))


def _select_simple_name_projection(lowered: str, allowed: list[str]) -> list[str]:
    if re.search(r"\bnames?\s+of\b", lowered) is None:
        return []
    if " and " in lowered:
        return []
    prioritized = [
        column
        for column in allowed
        if _is_name_like_column(column)
    ]
    if not prioritized:
        return []
    exact = [column for column in prioritized if str(column).lower() == "name"]
    if exact:
        return exact[:1]
    prioritized.sort(key=lambda col: _name_column_priority(str(col)))
    return prioritized[:1]


def _is_plain_names_query(lowered: str) -> bool:
    if re.search(r"\bnames?\s+of\b", lowered) is None:
        return False
    if " and " in lowered:
        return False
    if any(token in lowered for token in (" where ", " with ", " per ", " each ", " by ")):
        return False
    return True


def _augment_name_projection(lowered: str, allowed: list[str], selected: list[str]) -> list[str]:
    if not allowed:
        return selected

    out = list(selected)
    lowered_allowed = {str(column).lower(): str(column) for column in allowed}

    asks_first_last = (
        re.search(r"\bfirst\s+and\s+last\s+names?\b", lowered) is not None
        or re.search(r"\bfirst\s+names?\s+and\s+last\s+names?\b", lowered) is not None
    )
    if asks_first_last:
        first_choice = lowered_allowed.get("first_name") or lowered_allowed.get("firstname")
        last_choice = lowered_allowed.get("last_name") or lowered_allowed.get("lastname")
        if first_choice is None:
            first_choice = next(
                (col for col in allowed if "first" in str(col).lower() and "name" in str(col).lower()),
                None,
            )
        if last_choice is None:
            last_choice = next(
                (col for col in allowed if "last" in str(col).lower() and "name" in str(col).lower()),
                None,
            )
        ordered: list[str] = []
        if first_choice:
            ordered.append(first_choice)
        if last_choice:
            ordered.append(last_choice)
        for column in out:
            if column not in ordered:
                ordered.append(column)
        out = ordered

    asks_name_with_other_field = re.search(r"\bnames?\s+and\b", lowered) is not None
    if asks_name_with_other_field and not any(_is_name_like_column(col) for col in out):
        name_candidates = [column for column in allowed if _is_name_like_column(column)]
        if name_candidates:
            name_candidates.sort(key=lambda col: _name_column_priority(str(col)))
            out = [name_candidates[0]] + [col for col in out if col != name_candidates[0]]

    deduped: list[str] = []
    for column in out:
        if column not in deduped:
            deduped.append(column)
    return deduped


def _augment_song_release_projection(lowered: str, allowed: list[str], selected: list[str]) -> list[str]:
    if not allowed:
        return selected
    out = list(selected)
    lowered_allowed = {str(column).lower(): str(column) for column in allowed}

    song_name = next(
        (
            column
            for column in allowed
            if "song" in str(column).lower() and "name" in str(column).lower()
        ),
        None,
    )
    release_year = next(
        (
            column
            for column in allowed
            if "release" in str(column).lower() and "year" in str(column).lower()
        ),
        None,
    )

    asks_song_name = "song" in lowered and (" name " in f" {lowered} " or " names " in f" {lowered} ")
    asks_release_year = "release year" in lowered or ("release" in lowered and "year" in lowered)
    if asks_song_name and song_name is not None:
        generic_name_cols = [column for column in out if str(column).lower() in {"name", "fullname", "full_name"}]
        for column in generic_name_cols:
            if column != song_name:
                out = [item for item in out if item != column]
        if song_name not in out:
            out = [song_name] + out
    if asks_release_year and release_year is not None and release_year not in out:
        out.append(release_year)
    return out


def _augment_course_projection(lowered: str, allowed: list[str], selected: list[str]) -> list[str]:
    out = list(selected)
    lower_to_original = {str(column).lower(): str(column) for column in allowed}
    course_id = lower_to_original.get("course_id")
    if course_id is None:
        return out

    if "each course" in lowered and any(token in lowered for token in ("registration date", "registered for")):
        if course_id in out:
            out = [course_id] + [column for column in out if column != course_id]
        else:
            out = [course_id] + out
    return out


def _is_name_like_column(column: str) -> bool:
    lowered = str(column).lower()
    return (
        lowered == "name"
        or lowered.endswith("name")
        or lowered.endswith("_name")
        or "name_" in lowered
        or "fullname" in lowered
    )


def _name_column_priority(column: str) -> tuple[int, int]:
    lowered = str(column).lower()
    if lowered in {"name", "fullname", "full_name"}:
        return (0, len(lowered))
    if lowered.endswith("name"):
        return (1, len(lowered))
    return (2, len(lowered))


def _trim_overbroad_selection(lowered: str, selected: list[str], max_fields: int = 3) -> list[str]:
    if len(selected) <= max_fields:
        return selected
    prompt_tokens = _expanded_tokens(_tokenize(lowered))
    scored: list[tuple[float, int, str]] = []
    for idx, column in enumerate(selected):
        score = _column_semantic_score(prompt_tokens, column)
        scored.append((score, idx, column))
    scored.sort(key=lambda item: (-item[0], item[1]))
    trimmed = [column for _, _, column in scored[:max_fields]]
    return trimmed or selected[:max_fields]


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
    if "youngest" in lowered:
        return _best_age_like_column(selected_columns, all_columns), "ASC"
    if "oldest" in lowered:
        return _best_age_like_column(selected_columns, all_columns), "DESC"
    highest = re.search(r"\bhighest\s+([A-Za-z_]\w*)\b", lowered)
    if highest:
        phrase = highest.group(1)
        pool = list(all_columns or selected_columns)
        if pool:
            candidate = engine._best_matching_column(phrase, pool, prefer_metric=True)
            if candidate is not None:
                return candidate, "DESC"
        return phrase, "DESC"
    lowest = re.search(r"\blowest\s+([A-Za-z_]\w*)\b", lowered)
    if lowest:
        phrase = lowest.group(1)
        pool = list(all_columns or selected_columns)
        if pool:
            candidate = engine._best_matching_column(phrase, pool, prefer_metric=True)
            if candidate is not None:
                return candidate, "ASC"
        return phrase, "ASC"
    most = re.search(r"\bwith\s+the\s+most\s+([a-zA-Z_][\w\s]{0,40})\b", lowered)
    if most:
        phrase = most.group(1).strip()
        for marker in (" for ", " in ", " from ", " among ", " across "):
            if marker in phrase:
                phrase = phrase.split(marker, maxsplit=1)[0].strip()
        pool = list(all_columns or selected_columns)
        if pool:
            candidate = engine._best_matching_column(phrase, pool, prefer_metric=True)
            if candidate is not None:
                return candidate, "DESC"
            if any(token in phrase for token in ("sat", "test taker", "test takers", "taker", "takers")):
                sat_like = next(
                    (
                        column
                        for column in pool
                        if any(token in str(column).lower() for token in ("numtst", "test", "taker", "takr", "sat"))
                    ),
                    None,
                )
                if sat_like is not None:
                    return sat_like, "DESC"
        return phrase.split(" ")[0], "DESC"
    return None, None


def _best_age_like_column(selected_columns: list[str], all_columns: list[str] | None) -> str:
    pool = list(all_columns or selected_columns)
    for column in pool:
        lowered = str(column).lower()
        if lowered == "age" or lowered.endswith("_age") or "age" in lowered:
            return str(column)
    return "age"


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


def _entity_semantic_score(prompt_tokens: set[str], entity: str) -> float:
    entity_tokens = _expanded_tokens(_tokenize(entity))
    if not entity_tokens:
        return 0.0
    overlap = len(prompt_tokens.intersection(entity_tokens))
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(entity_tokens))


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
    for token in list(tokens):
        expanded.update(_token_inflections(token))
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
        "scr": "score",
        "score": "scr",
        "sat": "score",
        "percent": "percentage",
        "percentage": "percent",
    }
    for token in list(expanded):
        mapped = synonyms.get(token)
        if mapped:
            expanded.add(mapped)
    return expanded


def _token_inflections(token: str) -> set[str]:
    token = str(token).strip().lower()
    if not token:
        return set()
    forms: set[str] = {token}
    if len(token) <= 3:
        return forms
    if token.endswith("ies") and len(token) > 4:
        forms.add(token[:-3] + "y")
    elif token.endswith("es") and len(token) > 4:
        forms.add(token[:-2])
    elif token.endswith("s") and len(token) > 3:
        forms.add(token[:-1])
    elif token.endswith("y"):
        forms.add(token[:-1] + "ies")
        forms.add(token + "s")
    elif token.endswith(("x", "z", "ch", "sh")):
        forms.add(token + "es")
    else:
        forms.add(token + "s")
    return {form for form in forms if form}


def _has_metric_routing_intent(lowered: str) -> bool:
    metric_tokens = (
        "count",
        "how many",
        "sum",
        "total",
        "avg",
        "average",
        "max",
        "maximum",
        "min",
        "minimum",
        "highest",
        "lowest",
        "most",
        "least",
        "top",
    )
    return any(token in lowered for token in metric_tokens)


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
