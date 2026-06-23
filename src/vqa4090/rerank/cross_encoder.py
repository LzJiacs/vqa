from __future__ import annotations

from sentence_transformers import CrossEncoder

from vqa4090.utils.model_resolver import resolve_model_source


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        resolved, provider = resolve_model_source(model_name)
        print(f"[CrossEncoderReranker] model source: {provider} -> {resolved}")
        self.model = CrossEncoder(resolved)

    def score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [[query, d] for d in docs]
        scores = self.model.predict(pairs)
        return [float(x) for x in scores]
