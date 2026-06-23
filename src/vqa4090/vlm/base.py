from __future__ import annotations

from typing import Protocol


class VLMClient(Protocol):
    def answer(self, question: str, evidence_texts: list[str], image_paths: list[str] | None = None) -> str:
        ...


class MockVLM:
    def answer(self, question: str, evidence_texts: list[str], image_paths: list[str] | None = None) -> str:
        if not evidence_texts:
            return "I don't know."
        return f"Based on evidence: {evidence_texts[0][:180]}"
