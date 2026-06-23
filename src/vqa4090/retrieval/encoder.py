from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from vqa4090.utils.model_resolver import resolve_model_source


class DenseEncoder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        resolved, provider = resolve_model_source(model_name)
        print(f"[DenseEncoder] model source: {provider} -> {resolved}")
        self.model = SentenceTransformer(resolved)

    def encode(self, texts: list[str], normalize: bool = True, batch_size: int = 64) -> np.ndarray:
        vec = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype("float32")
