from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from text2ql.engines.graphql import GraphQLEngine
from text2ql.engines.sql import SQLEngine
from text2ql.providers.base import LLMProvider
from text2ql.providers.rule_based import RuleBasedProvider
from text2ql.types import QueryRequest, QueryResult


@dataclass(slots=True)
class Text2QL:
    """Main service facade for text-to-query conversion."""

    provider: LLMProvider = field(default_factory=RuleBasedProvider)
    _engines: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._engines = {
            "graphql": GraphQLEngine(provider=self.provider),
            "sql": SQLEngine(provider=self.provider),
        }

    def register_engine(self, name: str, engine: object) -> None:
        self._engines[name.lower()] = engine

    def generate(
        self,
        text: str,
        target: str = "graphql",
        schema: dict | None = None,
        mapping: dict | None = None,
        context: dict | None = None,
    ) -> QueryResult:
        normalized_target = target.lower().strip()
        if normalized_target not in self._engines:
            supported = ", ".join(sorted(self._engines))
            raise ValueError(f"Unsupported target '{target}'. Supported targets: {supported}")

        request = QueryRequest(
            text=text,
            target=normalized_target,
            schema=schema,
            mapping=mapping,
            context=context or {},
        )
        return self._engines[normalized_target].generate(request)
