from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from text2ql.filters import SPURIOUS_FILTER_VALUES as _DEFAULT_SPURIOUS_FILTER_VALUES


def contains_token(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def contains_entity_token(text: str, token: str) -> bool:
    normalized = str(token).strip().lower()
    if not normalized:
        return False

    for variant in label_match_variants(normalized):
        if contains_token(text, variant):
            return True

    if normalized.endswith("s"):
        return False

    for variant in label_match_variants(f"{normalized}s"):
        if contains_token(text, variant):
            return True
    return False


def sorted_alias_pairs(alias_map: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(alias_map.items(), key=lambda pair: len(pair[0]), reverse=True)


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def split_top_level(text: str, operator: str) -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    token = f" {operator} "
    token_len = len(token)
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and text[i : i + token_len] == token:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            i += token_len
            start = i
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def strip_outer_parentheses(text: str) -> str:
    candidate = text.strip()
    while candidate.startswith("(") and candidate.endswith(")"):
        depth = 0
        balanced = True
        for idx, ch in enumerate(candidate):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
                if depth == 0 and idx != len(candidate) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        candidate = candidate[1:-1].strip()
    return candidate


def extract_where_clause(lowered: str) -> str | None:
    parts = lowered.split(" where ", maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return None


def extract_filter_value(
    alias: str,
    text: str,
    *,
    spurious_values: Iterable[str] = _DEFAULT_SPURIOUS_FILTER_VALUES,
) -> str | None:
    key_pattern = re.escape(alias)
    patterns = [
        rf"\b{key_pattern}\b\s*(?:=|:)\s*([^\s,)\]]+)",
        rf"\b{key_pattern}\b\s+(?:is\s+|equals\s+|equal to\s+)?([^\s,)\]]+)",
    ]
    spurious = {str(item).lower() for item in spurious_values}
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if not matches:
            continue
        candidate = matches[-1].group(1).strip().strip('"').strip("'")
        if candidate.lower() in spurious:
            continue
        return candidate
    return None


def contains_column_reference(text: str, label: str) -> bool:
    for variant in label_match_variants(label):
        if contains_token(text, variant):
            return True
    return False


def label_match_variants(label: str) -> set[str]:
    lowered = str(label).strip().lower()
    if not lowered:
        return set()

    variants: set[str] = {lowered}
    normalized = re.sub(r"[_\s]+", " ", lowered)
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized:
        variants.add(normalized)

    tokens = [token for token in normalized.split(" ") if token]
    if len(tokens) == 1:
        variants.update(token_inflections(tokens[0]))

    if tokens:
        for replacement in token_inflections(tokens[-1]):
            phrase = " ".join([*tokens[:-1], replacement]).strip()
            if phrase:
                variants.add(phrase)

    return {variant for variant in variants if variant}


def token_inflections(token: str) -> set[str]:
    token = token.strip().lower()
    if not token:
        return set()
    forms: set[str] = {token}
    if len(token) > 3:
        if token.endswith("ies"):
            forms.add(token[:-3] + "y")
        elif token.endswith("es"):
            forms.add(token[:-2])
        elif token.endswith("s"):
            forms.add(token[:-1])
        elif token.endswith("y"):
            forms.add(token[:-1] + "ies")
            forms.add(token + "s")
        elif token.endswith(("x", "z", "ch", "sh")):
            forms.add(token + "es")
        else:
            forms.add(token + "s")
    return {form for form in forms if form}


def parse_grouped_boolean_filters(
    lowered: str,
    parse_and_nodes: Callable[[str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Parse AND/OR grouped boolean filters with parenthesis precedence."""
    if " and " not in lowered and " or " not in lowered:
        return {}
    where_clause = strip_outer_parentheses(extract_where_clause(lowered) or lowered)
    or_parts = split_top_level(where_clause, "or")
    if len(or_parts) <= 1:
        and_nodes = parse_and_nodes(where_clause)
        return {"and": and_nodes} if len(and_nodes) > 1 else {}
    nodes: list[dict[str, Any]] = []
    for part in or_parts:
        and_nodes = parse_and_nodes(part)
        if len(and_nodes) == 1:
            nodes.append(and_nodes[0])
        elif len(and_nodes) > 1:
            nodes.append({"and": and_nodes})
    return {"or": nodes} if len(nodes) > 1 else {}
