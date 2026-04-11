from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from text2ql.engines.graphql import GraphQLEngine
from text2ql.engines.sql import SQLEngine
from text2ql.providers.base import LLMProvider
from text2ql.providers.rule_based import RuleBasedProvider
from text2ql.types import QueryRequest, QueryResult

logger = logging.getLogger(__name__)


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

    def _make_request(
        self,
        text: str,
        target: str,
        schema: dict | None,
        mapping: dict | None,
        context: dict | None,
    ) -> tuple[str, QueryRequest]:
        normalized_target = target.lower().strip()
        if normalized_target not in self._engines:
            supported = ", ".join(sorted(self._engines))
            raise ValueError(f"Unsupported target '{target}'. Supported targets: {supported}")
        return normalized_target, QueryRequest(
            text=text,
            target=normalized_target,
            schema=schema,
            mapping=mapping,
            context=context or {},
        )

    def generate(
        self,
        text: str,
        target: str = "graphql",
        schema: dict | None = None,
        mapping: dict | None = None,
        context: dict | None = None,
    ) -> QueryResult:
        normalized_target, request = self._make_request(text, target, schema, mapping, context)
        logger.debug("Text2QL.generate: target=%r text=%r mode=%r", normalized_target, text, (context or {}).get("mode", "deterministic"))
        result = self._engines[normalized_target].generate(request)
        logger.debug("Text2QL.generate: query=%r confidence=%.2f", result.query, result.confidence or 0)
        return result

    async def agenerate(
        self,
        text: str,
        target: str = "graphql",
        schema: dict | None = None,
        mapping: dict | None = None,
        context: dict | None = None,
    ) -> QueryResult:
        normalized_target, request = self._make_request(text, target, schema, mapping, context)
        return await self._engines[normalized_target].agenerate(request)

    async def agenerate_many(
        self,
        requests: list[dict[str, Any]],
        concurrency: int = 5,
    ) -> list[QueryResult]:
        """Run multiple generate requests concurrently.

        Each entry in ``requests`` is a dict of kwargs accepted by ``agenerate``
        (text, target, schema, mapping, context). Results are returned in the
        same order as the input list.

        ``concurrency`` caps simultaneous in-flight requests — use a lower value
        when hitting rate-limited LLM providers.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _one(kw: dict[str, Any]) -> QueryResult:
            async with sem:
                return await self.agenerate(**kw)

        return list(await asyncio.gather(*[_one(kw) for kw in requests]))
