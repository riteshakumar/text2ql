from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Adapter interface for completion providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a raw model completion."""

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        """Async completion — default offloads sync complete to a thread pool."""
        return await asyncio.to_thread(self.complete, system_prompt, user_prompt)

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> str:
        """Return a model completion constrained to *json_schema*.

        The default implementation delegates to :meth:`complete` and ignores
        the schema — the response is validated downstream by the constrained
        parser.  Subclasses that support native structured output (e.g. OpenAI
        ``response_format: json_schema``) should override this method.

        Parameters
        ----------
        system_prompt:
            System-level instruction.
        user_prompt:
            User message.
        json_schema:
            A JSON Schema dict describing the expected output shape.  Passed
            as-is to the provider when the provider supports it.
        """
        logger.debug(
            "complete_structured: no structured-output support in %s; "
            "falling back to plain complete()",
            type(self).__name__,
        )
        return self.complete(system_prompt, user_prompt)

    async def acomplete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> str:
        """Async variant of :meth:`complete_structured`.

        Default offloads the sync version to a thread pool.  Override for
        truly non-blocking structured output.
        """
        return await asyncio.to_thread(
            self.complete_structured, system_prompt, user_prompt, json_schema
        )
