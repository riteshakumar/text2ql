from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from text2ql.providers.base import LLMProvider

RewritePlugin = Callable[["DatasetExample"], list[str]]


@dataclass(slots=True)
class DatasetExample:
    text: str
    target: str
    expected_query: str
    schema: dict[str, Any] | None = None
    mapping: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RewriteCandidate:
    text: str
    source: str
    confidence: float
    novelty: float = 0.0
    score: float = 0.0


def ingest_dataset(path: str | Path) -> list[DatasetExample]:
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".jsonl":
        return _ingest_jsonl(dataset_path)
    if dataset_path.suffix.lower() == ".json":
        return _ingest_json(dataset_path)
    raise ValueError("Unsupported dataset format. Use .json or .jsonl")


def generate_synthetic_examples(
    seed_examples: list[DatasetExample],
    variants_per_example: int = 1,
    provider: LLMProvider | None = None,
    rewrite_plugins: Sequence[str | RewritePlugin] | None = None,
    domain: str | None = None,
) -> list[DatasetExample]:
    synthetic: list[DatasetExample] = []
    resolved_plugins = _resolve_rewrite_plugins(rewrite_plugins)
    for example in seed_examples:
        requested = max(0, variants_per_example)
        if requested == 0:
            continue
        inferred_domain = (domain or _infer_example_domain(example) or "").strip().lower() or None
        example_plugins = _select_plugins_for_example(example, resolved_plugins, inferred_domain)
        candidates = _collect_rewrite_candidates(example, example_plugins, inferred_domain)
        ranked = _rank_candidates(example.text, candidates)
        selected = _select_ranked_candidates(ranked, requested)
        for i, candidate in enumerate(selected, start=1):
            synthetic.append(
                DatasetExample(
                    text=candidate.text,
                    target=example.target,
                    expected_query=example.expected_query,
                    schema=example.schema,
                    mapping=example.mapping,
                    context={**example.context, "synthetic": True},
                    metadata={
                        **example.metadata,
                        "synthetic_variant": i,
                        "synthetic_domain": inferred_domain,
                        "synthetic_rewrite_source": candidate.source,
                        "synthetic_rewrite_confidence": candidate.confidence,
                        "synthetic_rewrite_novelty": candidate.novelty,
                        "synthetic_rewrite_score": candidate.score,
                    },
                )
            )

            if provider is not None:
                hook_payload = {
                    "seed_text": example.text,
                    "synthetic_text": candidate.text,
                    "target": example.target,
                }
                try:
                    provider.complete(
                        "Return a concise quality note for synthetic query generation.",
                        json.dumps(hook_payload),
                    )
                except (RuntimeError, ValueError, TypeError):
                    # Hook is optional and should never block generation.
                    pass
    return synthetic


def _ingest_jsonl(path: Path) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(_parse_example(payload))
    return examples


def _ingest_json(path: Path) -> list[DatasetExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON dataset must be a list of examples")
    return [_parse_example(item) for item in payload]


def _parse_example(payload: dict[str, Any]) -> DatasetExample:
    if not isinstance(payload, dict):
        raise ValueError("Each dataset item must be an object")

    text = payload.get("text")
    expected_query = payload.get("expected_query")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Example field 'text' is required and must be a string")
    if not isinstance(expected_query, str) or not expected_query.strip():
        raise ValueError("Example field 'expected_query' is required and must be a string")

    target = payload.get("target", "graphql")
    schema = payload.get("schema")
    mapping = payload.get("mapping")
    context = payload.get("context", {})
    metadata = payload.get("metadata", {})

    if not isinstance(target, str):
        raise ValueError("Example field 'target' must be a string")
    if schema is not None and not isinstance(schema, dict):
        raise ValueError("Example field 'schema' must be an object when provided")
    if mapping is not None and not isinstance(mapping, dict):
        raise ValueError("Example field 'mapping' must be an object when provided")
    if not isinstance(context, dict):
        raise ValueError("Example field 'context' must be an object")
    if not isinstance(metadata, dict):
        raise ValueError("Example field 'metadata' must be an object")

    return DatasetExample(
        text=text,
        target=target,
        expected_query=expected_query,
        schema=schema,
        mapping=mapping,
        context=context,
        metadata=metadata,
    )


def _resolve_rewrite_plugins(
    rewrite_plugins: Sequence[str | RewritePlugin] | None,
) -> list[tuple[str, RewritePlugin]]:
    if rewrite_plugins is None:
        return [("generic", _generic_rewrite_plugin)]
    resolved: list[tuple[str, RewritePlugin]] = []
    for plugin in rewrite_plugins:
        if isinstance(plugin, str):
            name = plugin.strip().lower()
            if name not in BUILTIN_REWRITE_PLUGINS:
                supported = ", ".join(sorted(BUILTIN_REWRITE_PLUGINS))
                raise ValueError(f"Unknown rewrite plugin '{plugin}'. Supported plugins: {supported}")
            resolved.append((name, BUILTIN_REWRITE_PLUGINS[name]))
            continue
        if not callable(plugin):
            raise ValueError("rewrite_plugins entries must be plugin names or callable plugins")
        resolved.append((getattr(plugin, "__name__", "custom"), plugin))
    if not resolved:
        return [("generic", _generic_rewrite_plugin)]
    return resolved


def _select_plugins_for_example(
    example: DatasetExample,
    configured_plugins: list[tuple[str, RewritePlugin]],
    domain: str | None,
) -> list[tuple[str, RewritePlugin]]:
    selected = list(configured_plugins)
    inferred_domain = (domain or _infer_example_domain(example) or "").strip().lower()
    if not inferred_domain:
        return selected
    domain_plugin = BUILTIN_DOMAIN_PLUGINS.get(inferred_domain)
    if domain_plugin is None:
        return selected
    if not any(name == inferred_domain for name, _ in selected):
        selected.append((inferred_domain, domain_plugin))
    return selected


def _collect_rewrite_candidates(
    example: DatasetExample,
    plugins: list[tuple[str, RewritePlugin]],
    domain: str | None,
) -> list[_RewriteCandidate]:
    seen: set[str] = set()
    ordered: list[_RewriteCandidate] = []
    allowed_lexicon = _build_allowed_lexicon(example)
    for plugin_name, plugin in plugins:
        try:
            outputs = plugin(example)
        except Exception:  # noqa: BLE001
            continue
        for candidate in outputs:
            _append_candidate(
                ordered=ordered,
                seen=seen,
                text=candidate,
                source=plugin_name,
                allowed_lexicon=allowed_lexicon,
            )

    for template in _domain_template_rewrites(example, domain):
        source_name = f"{domain}-template" if domain else "template"
        _append_candidate(
            ordered=ordered,
            seen=seen,
            text=template,
            source=source_name,
            allowed_lexicon=allowed_lexicon,
        )

    seed_text = example.text.strip() or example.text
    if seed_text and seed_text not in seen:
        ordered.append(
            _RewriteCandidate(
                text=seed_text,
                source="seed",
                confidence=_source_confidence("seed"),
            )
        )
    return ordered or [_RewriteCandidate(text=example.text, source="seed", confidence=0.35)]


def _append_candidate(
    ordered: list[_RewriteCandidate],
    seen: set[str],
    text: str,
    source: str,
    allowed_lexicon: set[str],
) -> None:
    cleaned = text.strip()
    if not cleaned or cleaned in seen:
        return
    if allowed_lexicon and not _is_schema_lexically_valid(cleaned, allowed_lexicon):
        return
    seen.add(cleaned)
    ordered.append(
        _RewriteCandidate(
            text=cleaned,
            source=source,
            confidence=_source_confidence(source),
        )
    )


def _rank_candidates(seed_text: str, candidates: list[_RewriteCandidate]) -> list[_RewriteCandidate]:
    ranked: list[_RewriteCandidate] = []
    for candidate in candidates:
        novelty = _token_novelty(seed_text, candidate.text)
        score = round((0.6 * candidate.confidence) + (0.4 * novelty), 6)
        ranked.append(
            _RewriteCandidate(
                text=candidate.text,
                source=candidate.source,
                confidence=candidate.confidence,
                novelty=novelty,
                score=score,
            )
        )
    ranked.sort(key=lambda item: (item.score, item.novelty, len(item.text)), reverse=True)
    return ranked


def _select_ranked_candidates(ranked: list[_RewriteCandidate], count: int) -> list[_RewriteCandidate]:
    if count <= 0 or not ranked:
        return []
    if count <= len(ranked):
        return ranked[:count]
    selected: list[_RewriteCandidate] = []
    for index in range(count):
        selected.append(ranked[index % len(ranked)])
    return selected


def _source_confidence(source: str) -> float:
    if source.endswith("-template") or source == "template":
        return 0.64
    if source == "seed":
        return 0.35
    if source == "generic":
        return 0.72
    return 0.88


def _token_novelty(seed_text: str, candidate_text: str) -> float:
    seed_tokens = _tokens(seed_text)
    candidate_tokens = _tokens(candidate_text)
    if not seed_tokens and not candidate_tokens:
        return 0.0
    union = seed_tokens.union(candidate_tokens)
    if not union:
        return 0.0
    intersection = seed_tokens.intersection(candidate_tokens)
    return round(1.0 - (len(intersection) / len(union)), 6)


def _build_allowed_lexicon(example: DatasetExample) -> set[str]:
    terms: set[str] = set()
    for source in (example.schema, example.mapping):
        if not isinstance(source, dict):
            continue
        for value in _iter_strings(source):
            terms.update(_tokens(value))
    return terms


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(nested)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _tokens(text: str) -> set[str]:
    with_spaces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    normalized = with_spaces.replace("_", " ").lower()
    return {token for token in re.findall(r"[a-z0-9]+", normalized) if token}


def _is_schema_lexically_valid(text: str, allowed_lexicon: set[str]) -> bool:
    content_tokens = [
        token
        for token in _tokens(text)
        if token not in _NEUTRAL_TOKENS and len(token) > 2 and not token.isdigit()
    ]
    if not content_tokens:
        return True
    unknown = [
        token
        for token in content_tokens
        if token not in allowed_lexicon and token.rstrip("s") not in allowed_lexicon
    ]
    allowed_unknown = max(1, len(content_tokens) // 3)
    return len(unknown) <= allowed_unknown


def _domain_template_rewrites(example: DatasetExample, domain: str | None) -> list[str]:
    if not domain:
        return []
    lowered = example.text.lower()
    # For strong intent prompts like "how many <asset> do I own", template rewrites
    # tend to drift; rely on domain plugins instead.
    if _match_owned_asset_phrase(lowered) is not None:
        return []
    if "how many" in lowered and ("own" in lowered or "shares" in lowered):
        return []
    slots = _slot_catalog(example)
    if not _domain_slots_compatible(domain, slots):
        return []
    entities = slots["entity"][:2]

    rewrites: list[str] = []
    for entity in entities:
        entity_metrics = _entity_metrics(example.schema, entity)[:2]
        entity_dates = _entity_dates(example.schema, entity)[:2]
        entity_filters = _entity_filters(example.schema, entity)[:2]
        entity_values = _filter_values_for_keys(example.mapping, entity_filters)[:2]

        if entity_metrics:
            rewrites.append(f"show {entity} with {entity_metrics[0]}")
            rewrites.append(f"what is {entity_metrics[0]} for {entity}")
        if entity_dates:
            rewrites.append(f"show {entity} by {entity_dates[0]}")
        if entity_filters and entity_values:
            rewrites.append(f"show {entity} where {entity_filters[0]} is {entity_values[0]}")
            rewrites.append(f"list {entity} with {entity_filters[0]} {entity_values[0]}")
    return rewrites


def _domain_slots_compatible(domain: str, slots: dict[str, list[str]]) -> bool:
    hints = _DOMAIN_SLOT_HINTS.get(domain.lower())
    if not hints:
        return True
    combined = " ".join(
        slots.get("entity", [])
        + slots.get("metric", [])
        + slots.get("date", [])
        + slots.get("filter", [])
        + slots.get("value", [])
    ).lower()
    return any(hint in combined for hint in hints)


def _slot_catalog(example: DatasetExample) -> dict[str, list[str]]:
    entities = _extract_entities(example.schema)
    fields = _extract_fields(example.schema)
    args = _extract_args(example.schema)
    values = _extract_filter_values(example.mapping)

    metric_fields = [
        field
        for field in fields
        if any(token in field.lower() for token in _METRIC_HINTS)
    ]
    date_fields = [
        field
        for field in fields
        if any(token in field.lower() for token in _DATE_HINTS)
    ]
    filter_fields = [
        field
        for field in (args + fields)
        if any(token in field.lower() for token in _FILTER_HINTS)
    ]

    return {
        "entity": _dedupe_preserve_order(entities),
        "metric": _dedupe_preserve_order(metric_fields),
        "date": _dedupe_preserve_order(date_fields),
        "filter": _dedupe_preserve_order(filter_fields),
        "value": _dedupe_preserve_order(values),
    }


def _extract_entities(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    entities_raw = schema.get("entities")
    out: list[str] = []
    if isinstance(entities_raw, list):
        for item in entities_raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str):
                    out.append(name)
    return out


def _extract_fields(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    fields_raw = schema.get("fields")
    if isinstance(fields_raw, dict):
        return [
            item
            for values in fields_raw.values()
            if isinstance(values, list)
            for item in values
            if isinstance(item, str)
        ]
    if isinstance(fields_raw, list):
        return _extract_named_or_string_list(fields_raw)
    return []


def _extract_args(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    args_raw = schema.get("args")
    out: list[str] = []
    if isinstance(args_raw, dict):
        for values in args_raw.values():
            if isinstance(values, list):
                out.extend([item for item in values if isinstance(item, str)])
    elif isinstance(args_raw, list):
        out.extend([item for item in args_raw if isinstance(item, str)])
    return out


def _extract_filter_values(mapping: dict[str, Any] | None) -> list[str]:
    if not isinstance(mapping, dict):
        return []
    candidates: list[str] = []
    for key in ("filter_values", "filter_value_aliases"):
        candidates.extend(_extract_nested_str_values(mapping.get(key)))
    return candidates


def _entity_metrics(schema: dict[str, Any] | None, entity: str) -> list[str]:
    fields = _extract_fields_for_entity(schema, entity)
    return [field for field in fields if any(token in field.lower() for token in _METRIC_HINTS)]


def _entity_dates(schema: dict[str, Any] | None, entity: str) -> list[str]:
    fields = _extract_fields_for_entity(schema, entity)
    return [field for field in fields if any(token in field.lower() for token in _DATE_HINTS)]


def _entity_filters(schema: dict[str, Any] | None, entity: str) -> list[str]:
    fields = _extract_fields_for_entity(schema, entity)
    args = _extract_args_for_entity(schema, entity)
    return [
        field
        for field in (args + fields)
        if any(token in field.lower() for token in _FILTER_HINTS)
    ]


def _extract_fields_for_entity(schema: dict[str, Any] | None, entity: str) -> list[str]:
    if not isinstance(schema, dict):
        return []
    fields_raw = schema.get("fields")
    if isinstance(fields_raw, dict):
        values = fields_raw.get(entity, [])
        if isinstance(values, list):
            return [item for item in values if isinstance(item, str)]
        return []
    return _extract_fields(schema)


def _extract_args_for_entity(schema: dict[str, Any] | None, entity: str) -> list[str]:
    if not isinstance(schema, dict):
        return []
    args_raw = schema.get("args")
    if isinstance(args_raw, dict):
        values = args_raw.get(entity, [])
        if isinstance(values, list):
            return [item for item in values if isinstance(item, str)]
        return []
    return _extract_args(schema)


def _filter_values_for_keys(mapping: dict[str, Any] | None, keys: list[str]) -> list[str]:
    if not isinstance(mapping, dict) or not keys:
        return []
    out: list[str] = []
    for bucket in ("filter_values", "filter_value_aliases"):
        payload = mapping.get(bucket)
        if not isinstance(payload, dict):
            continue
        for key in keys:
            nested = payload.get(key)
            if not isinstance(nested, dict):
                continue
            out.extend(value for value in nested.values() if isinstance(value, str))
    return _dedupe_preserve_order(out)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _infer_example_domain(example: DatasetExample) -> str | None:
    lowered_text = example.text.lower()
    text_domain = _match_domain_by_keywords(lowered_text, _DOMAIN_TEXT_HINTS)
    if text_domain is not None:
        return text_domain
    if isinstance(example.schema, dict):
        schema_blob = json.dumps(example.schema, sort_keys=True).lower()
        return _match_domain_by_keywords(schema_blob, _DOMAIN_SCHEMA_HINTS)
    return None


def _extract_named_or_string_list(items: list[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
            continue
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str):
                out.append(name)
    return out


def _extract_nested_str_values(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    out: list[str] = []
    for nested in value.values():
        if not isinstance(nested, dict):
            continue
        out.extend(str_value for str_value in nested.values() if isinstance(str_value, str))
    return out


def _match_domain_by_keywords(content: str, domain_hints: dict[str, tuple[str, ...]]) -> str | None:
    for domain, hints in domain_hints.items():
        if any(token in content for token in hints):
            return domain
    return None


def _generic_rewrite_plugin(example: DatasetExample) -> list[str]:
    text = example.text
    rewrites = [
        text.replace("show", "list"),
        text.replace("list", "show"),
        text.replace("top", "first"),
        text.replace("first", "top"),
        text.replace("what is", "show"),
    ]
    return [candidate for candidate in rewrites if candidate.strip()]


def _portfolio_rewrite_plugin(example: DatasetExample) -> list[str]:
    text = example.text
    lowered = text.lower()
    rewrites: list[str] = []

    match = _match_owned_asset_phrase(lowered)
    if match is not None:
        asset = match.upper()
        rewrites.extend(
            [
                f"what quantity of {asset} do i own",
                f"how many shares of {asset} are in my portfolio",
                f"show my {asset} position quantity",
            ]
        )

    if "total market value" in lowered:
        rewrites.extend(
            [
                "show my portfolio total market value",
                "what is the account market value total",
            ]
        )
    if "dividend" in lowered:
        rewrites.append("show dividend transactions")
    if "buying power" in lowered:
        rewrites.append("show cash buying power details")

    return rewrites


def _banking_rewrite_plugin(example: DatasetExample) -> list[str]:
    lowered = example.text.lower()
    rewrites: list[str] = []

    if "balance" in lowered:
        rewrites.extend(
            [
                "show current account balance",
                "what is my available balance",
            ]
        )
    if "transfer" in lowered:
        rewrites.extend(
            [
                "show recent transfer activity",
                "list transfer transactions",
            ]
        )
    if "deposit" in lowered:
        rewrites.append("show deposit transactions")
    if "withdraw" in lowered:
        rewrites.append("show withdrawal transactions")
    if "statement" in lowered:
        rewrites.append("list statement entries")

    return rewrites


def _ecommerce_rewrite_plugin(example: DatasetExample) -> list[str]:
    lowered = example.text.lower()
    rewrites: list[str] = []

    if "order" in lowered:
        rewrites.extend(
            [
                "list customer orders",
                "show recent order records",
            ]
        )
    if "cart" in lowered:
        rewrites.append("show active shopping cart items")
    if "product" in lowered or "sku" in lowered:
        rewrites.extend(
            [
                "list product catalog items",
                "show product sku details",
            ]
        )
    if "inventory" in lowered:
        rewrites.append("show inventory stock levels")
    if "refund" in lowered:
        rewrites.append("list refund transactions")
    if "shipment" in lowered:
        rewrites.append("show shipment status records")

    return rewrites


def _crm_rewrite_plugin(example: DatasetExample) -> list[str]:
    lowered = example.text.lower()
    rewrites: list[str] = []

    if "lead" in lowered:
        rewrites.extend(
            [
                "list open leads",
                "show lead records",
            ]
        )
    if "opportunity" in lowered or "pipeline" in lowered:
        rewrites.extend(
            [
                "show sales opportunities in pipeline",
                "list pipeline opportunity records",
            ]
        )
    if "contact" in lowered:
        rewrites.append("show contact details")
    if "deal" in lowered or "stage" in lowered:
        rewrites.append("list deals by sales stage")

    return rewrites


def _healthcare_rewrite_plugin(example: DatasetExample) -> list[str]:
    lowered = example.text.lower()
    rewrites: list[str] = []

    if "patient" in lowered:
        rewrites.extend(
            [
                "show patient records",
                "list patient details",
            ]
        )
    if "encounter" in lowered or "visit" in lowered:
        rewrites.append("show recent patient encounters")
    if "diagnosis" in lowered:
        rewrites.append("list diagnosis entries")
    if "medication" in lowered:
        rewrites.append("show patient medications")
    if "lab" in lowered:
        rewrites.append("show lab result records")
    if "claim" in lowered:
        rewrites.append("list insurance claim records")

    return rewrites


def _match_owned_asset_phrase(lowered: str) -> str | None:
    match = re.search(r"\bhow many\s+([a-z0-9_]+)\s+do i own\b", lowered)
    if match:
        return match.group(1)
    match = re.search(r"\bhow many\s+([a-z0-9_]+)\s+i own\b", lowered)
    if match:
        return match.group(1)
    return None


_NEUTRAL_TOKENS = {
    "show",
    "list",
    "what",
    "with",
    "where",
    "for",
    "and",
    "or",
    "the",
    "a",
    "an",
    "my",
    "is",
    "are",
    "of",
    "by",
    "in",
}

_METRIC_HINTS = (
    "amount",
    "total",
    "value",
    "price",
    "balance",
    "quantity",
    "count",
    "sum",
    "revenue",
    "cost",
    "score",
    "net",
    "gain",
    "loss",
)

_DATE_HINTS = (
    "date",
    "time",
    "created",
    "updated",
    "asof",
    "month",
    "year",
    "day",
)

_FILTER_HINTS = (
    "status",
    "type",
    "category",
    "stage",
    "state",
    "symbol",
    "region",
    "segment",
    "priority",
)

_DOMAIN_TEXT_HINTS: dict[str, tuple[str, ...]] = {
    "portfolio": ("portfolio", "positions", "holdings", "symbol", "account"),
    "banking": ("balance", "transfer", "deposit", "withdraw", "statement", "checking", "savings", "transactions"),
    "crm": ("lead", "opportunity", "pipeline", "contact", "deal", "account executive", "sales stage", "crm"),
    "healthcare": ("patient", "encounter", "diagnosis", "medication", "lab", "claim", "provider", "healthcare", "clinical"),
    "ecommerce": ("order", "cart", "checkout", "product", "sku", "inventory", "refund", "shipment"),
}

_DOMAIN_SCHEMA_HINTS: dict[str, tuple[str, ...]] = {
    "portfolio": ("positions", "transactions", "acct", "portfolio"),
    "banking": ("balance", "transfer", "deposit", "checking", "savings"),
    "crm": ("leads", "opportunities", "pipeline", "contacts", "crm"),
    "healthcare": ("patients", "encounters", "medications", "labs", "claims"),
    "ecommerce": ("orders", "products", "cart", "inventory", "shipments"),
}

_DOMAIN_SLOT_HINTS: dict[str, tuple[str, ...]] = {
    "portfolio": ("position", "holding", "symbol", "account", "market", "quantity"),
    "banking": ("balance", "transfer", "deposit", "withdraw", "checking", "savings", "statement"),
    "ecommerce": ("order", "product", "sku", "inventory", "cart", "refund", "shipment"),
    "crm": ("lead", "opportun", "pipeline", "contact", "deal", "stage"),
    "healthcare": ("patient", "encounter", "diagnos", "medication", "lab", "claim", "clinical"),
}


BUILTIN_REWRITE_PLUGINS: dict[str, RewritePlugin] = {
    "generic": _generic_rewrite_plugin,
    "portfolio": _portfolio_rewrite_plugin,
    "banking": _banking_rewrite_plugin,
    "crm": _crm_rewrite_plugin,
    "healthcare": _healthcare_rewrite_plugin,
    "ecommerce": _ecommerce_rewrite_plugin,
}

BUILTIN_DOMAIN_PLUGINS: dict[str, RewritePlugin] = {
    "portfolio": _portfolio_rewrite_plugin,
    "banking": _banking_rewrite_plugin,
    "crm": _crm_rewrite_plugin,
    "healthcare": _healthcare_rewrite_plugin,
    "ecommerce": _ecommerce_rewrite_plugin,
}
