from __future__ import annotations

import json
import re
from typing import Any

from text2ql.providers.base import LLMProvider
from text2ql.schema_config import normalize_schema_config

_IDENTIFIER_HINTS = ("symbol", "ticker", "security", "asset", "instrument", "code", "name")
_QUANTITY_HINTS = ("quantity", "shares", "units", "holding", "position")
_OWNERSHIP_HINTS = ("own", "hold", "holding", "shares", "quantity")


def rewrite_user_utterance(
    text: str,
    target: str,
    schema: dict[str, Any] | None,
    mapping: dict[str, Any] | None,
    provider: LLMProvider | None,
    system_context: str = "",
) -> tuple[str, dict[str, Any]]:
    if provider is None:
        return text, {"applied": False, "reason": "provider_unavailable"}

    config = normalize_schema_config(schema, mapping)
    owned_asset = _match_owned_asset_phrase(text.lower())
    quantity_label = _preferred_quantity_label(config)
    if owned_asset is not None and _schema_supports_identifier_and_quantity(config):
        canonical = f"how many {quantity_label} of {owned_asset.upper()} do i own"
        return canonical, {
            "applied": canonical.strip() != text.strip(),
            "source": "schema_slot_canonicalizer",
            "confidence": 1.0,
            "notes": "Canonicalized owned-asset quantity intent using schema-derived slot hints.",
            "raw": "",
        }
    inferred_asset = _match_how_many_asset_phrase(text.lower())
    if (
        inferred_asset is not None
        and _schema_supports_identifier_and_quantity(config)
        and _looks_like_identifier_value(inferred_asset, config)
    ):
        canonical = f"how many {quantity_label} of {inferred_asset.upper()} do i own"
        return canonical, {
            "applied": canonical.strip() != text.strip(),
            "source": "schema_slot_canonicalizer",
            "confidence": 0.95,
            "notes": "Canonicalized 'how many <asset>' intent into ownership quantity query using schema/mapping values.",
            "raw": "",
        }
    system_prompt = _build_system_prompt(target, system_context)
    user_prompt = _build_user_prompt(text=text, config=config, target=target)
    try:
        raw = provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    except (RuntimeError, ValueError, TypeError) as exc:
        return text, {"applied": False, "reason": f"provider_error: {exc}"}

    payload = _load_json_payload(raw)
    if not isinstance(payload, dict):
        return text, {"applied": False, "reason": "rewrite_parse_error", "raw": raw}
    rewritten = payload.get("rewritten_text")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return text, {"applied": False, "reason": "missing_rewritten_text", "raw": raw}
    if _looks_like_query_text(rewritten):
        return text, {
            "applied": False,
            "reason": "guard_rejected_query_like_rewrite",
            "notes": "Rewrite must remain natural language and not emit SQL/GraphQL query text.",
            "raw": raw,
        }
    if _has_ownership_intent(text) and not _has_ownership_intent(rewritten):
        return text, {
            "applied": False,
            "reason": "guard_rejected_intent_drift",
            "notes": "Original prompt expressed ownership intent; rewritten prompt removed it.",
            "raw": raw,
        }

    return rewritten.strip(), {
        "applied": rewritten.strip() != text.strip(),
        "source": "llm_rewrite",
        "confidence": payload.get("confidence"),
        "notes": payload.get("notes", ""),
        "raw": raw,
    }


def _build_system_prompt(target: str, system_context: str) -> str:
    base = (
        "You rewrite user utterances into clear canonical requests for text2ql.\n"
        "Return ONLY JSON object with keys: rewritten_text (string), notes (string), confidence (number 0..1).\n"
        "Do not invent entities/fields that are not present in schema context.\n"
        "Keep intent unchanged.\n"
        "Never convert concrete business questions into meta/help wording (e.g., avoid 'how to query ...').\n"
        "Never output SQL or GraphQL query text; rewritten_text must be natural-language."
    )
    if system_context.strip():
        return f"{base}\n\nAdditional system context:\n{system_context.strip()}"
    return base


def _build_user_prompt(text: str, config: Any, target: str) -> str:
    entities = config.entities
    fields = config.fields_by_entity or {"_": config.fields}
    args = config.args_by_entity
    filter_aliases = config.filter_key_aliases
    filter_values = config.filter_value_aliases
    return (
        f"Target: {target}\n"
        f"Original utterance: {text}\n\n"
        f"Entities: {json.dumps(entities)}\n"
        f"Fields by entity: {json.dumps(fields)}\n"
        f"Args by entity: {json.dumps(args)}\n"
        f"Filter aliases: {json.dumps(filter_aliases)}\n"
        f"Filter values: {json.dumps(filter_values)}\n"
    )


def _load_json_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload
    fenced = _extract_fenced_json(raw)
    if fenced is not None:
        try:
            payload = json.loads(fenced)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    blob = _extract_first_json_object(raw)
    if blob is not None:
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    return None


def _extract_fenced_json(raw: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.S | re.I)
    if not match:
        return None
    return match.group(1).strip()


def _extract_first_json_object(raw: str) -> str | None:
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1].strip()
    return None


def _match_owned_asset_phrase(lowered: str) -> str | None:
    match = re.search(r"\bhow many\s+([a-z0-9_]+)\s+do i own\b", lowered)
    if match:
        return match.group(1)
    match = re.search(r"\bhow many\s+([a-z0-9_]+)\s+i own\b", lowered)
    if match:
        return match.group(1)
    match = re.search(r"\bquantity of\s+([a-z0-9_]+)\s+do i own\b", lowered)
    if match:
        return match.group(1)
    return None


def _match_how_many_asset_phrase(lowered: str) -> str | None:
    match = re.search(r"\bhow many\s+([a-z0-9_]+)\b", lowered)
    if not match:
        return None
    asset = match.group(1)
    if asset in {"positions", "holdings", "accounts", "orders", "customers", "users", "items"}:
        return None
    return asset


def _has_ownership_intent(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _OWNERSHIP_HINTS)


def _schema_supports_identifier_and_quantity(config: Any) -> bool:
    fields: set[str] = set()
    for values in config.fields_by_entity.values():
        fields.update(str(item).lower() for item in values)
    for values in config.args_by_entity.values():
        fields.update(str(item).lower() for item in values)
    has_identifier = any(any(hint in field for hint in _IDENTIFIER_HINTS) for field in fields)
    has_quantity = any(any(hint in field for hint in _QUANTITY_HINTS) for field in fields)
    return has_identifier and has_quantity


def _preferred_quantity_label(config: Any) -> str:
    names: list[str] = []
    for values in config.fields_by_entity.values():
        names.extend(str(item).lower() for item in values)
    for values in config.args_by_entity.values():
        names.extend(str(item).lower() for item in values)
    for preferred in ("shares", "quantity", "units"):
        if any(preferred in name for name in names):
            return preferred
    return "quantity"


def _looks_like_identifier_value(asset: str, config: Any) -> bool:
    token = asset.strip().lower()
    if not token:
        return False
    for alias_map in config.filter_value_aliases.values():
        if not isinstance(alias_map, dict):
            continue
        for key, value in alias_map.items():
            if isinstance(key, str) and key.lower() == token:
                return True
            if isinstance(value, str) and value.lower() == token:
                return True
    return len(token) <= 5 and token.isalpha()


def _looks_like_query_text(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith("select ") or lowered.startswith("with "):
        return True
    if lowered.startswith("query ") or lowered.startswith("mutation "):
        return True
    if " from " in lowered and (" where " in lowered or lowered.startswith("select ")):
        return True
    if ("{" in stripped and "}" in stripped) and (
        lowered.startswith("query") or lowered.startswith("mutation") or "(" in stripped
    ):
        return True
    return False
