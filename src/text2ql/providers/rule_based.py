from __future__ import annotations

from .base import LLMProvider


class RuleBasedProvider(LLMProvider):
    """Deterministic fallback provider for local/offline development."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return user_prompt
