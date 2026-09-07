"""Select whole related tables for prompts without truncating their columns."""
from __future__ import annotations

from copy import copy
from collections import deque
import re

from text2ql.schema_config import NormalizedSchemaConfig


def select_prompt_schema(text: str, config: NormalizedSchemaConfig, max_entities: int = 50) -> NormalizedSchemaConfig:
    if len(config.entities) <= max_entities:
        return config
    if max_entities < 1:
        raise ValueError("max_entities must be positive")
    tokens = set(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()))
    scores: dict[str, int] = {}
    explicit: set[str] = set()
    for entity in config.entities:
        terms = [entity] + [alias for alias, canonical in config.entity_aliases.items() if canonical == entity]
        if any(re.search(rf"\b{re.escape(term.lower())}\b", text.lower()) for term in terms):
            explicit.add(entity)
        columns = config.fields_by_entity.get(entity, config.fields)
        words = set(re.findall(r"[a-z0-9]+", " ".join(columns + terms).lower()))
        scores[entity] = len(tokens & words) + 100 * (entity in explicit)
    if not any(scores.values()):
        raise ValueError("No relevant tables found in this large schema; name a table or provide aliases")
    selected = set(explicit)
    if not selected:
        selected.add(max(scores, key=scores.get))
    # Include shortest relationship paths between explicitly referenced tables.
    graph: dict[str, set[str]] = {name: set() for name in config.entities}
    for source, relations in config.relations_by_entity.items():
        for relation in relations.values():
            if source in graph and relation.target in graph:
                graph[source].add(relation.target)
                graph[relation.target].add(source)
    for start in sorted(selected):
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            current, path = queue.popleft()
            if current in explicit:
                selected.update(path)
            for neighbor in sorted(graph[current] - seen):
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    if len(selected) > max_entities:
        raise ValueError("The requested relationship path exceeds the prompt table budget")
    for name in sorted(config.entities, key=lambda item: (-scores[item], item)):
        if len(selected) >= max_entities:
            break
        if scores[name] > 0 or any(name in graph[chosen] for chosen in selected):
            selected.add(name)
    result = copy(config)
    result.entities = [name for name in config.entities if name in selected]
    result.args_by_entity = {name: args for name, args in config.args_by_entity.items() if name in selected}
    result.introspection_query_args = {name: args for name, args in config.introspection_query_args.items() if name in selected}
    result.fields_by_entity = {name: list(config.fields_by_entity.get(name, config.fields)) for name in result.entities}
    result.fields = list(dict.fromkeys(field for fields in result.fields_by_entity.values() for field in fields))
    result.relations_by_entity = {name: {key: rel for key, rel in config.relations_by_entity.get(name, {}).items() if rel.target in selected} for name in result.entities}
    return result
