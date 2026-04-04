from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Adapter interface for completion providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a raw model completion."""
