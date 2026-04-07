from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Adapter interface for completion providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a raw model completion."""

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        """Async completion — default offloads sync complete to a thread pool."""
        return await asyncio.to_thread(self.complete, system_prompt, user_prompt)
