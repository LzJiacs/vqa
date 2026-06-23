from __future__ import annotations

import os
from pathlib import Path


def resolve_model_source(model_name_or_path: str, cache_dir: str | None = None) -> tuple[str, str]:
    """
    Resolve model source with priority:
    1) local path
    2) ModelScope snapshot_download (if available)
    3) fallback to original model id (for HuggingFace loaders)
    Returns: (resolved_path_or_id, provider)
    """
    raw = str(model_name_or_path).strip()
    if not raw:
        raise ValueError("model_name_or_path is empty")

    p = Path(raw)
    if p.exists():
        return str(p), "local"

    prefer_modelscope = os.getenv("VQA_PREFER_MODELSCOPE", "1").strip() not in {"0", "false", "False"}
    if prefer_modelscope:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            ms_cache = cache_dir or os.getenv("MODELSCOPE_CACHE", str(Path.home() / ".cache" / "modelscope"))
            local_dir = snapshot_download(model_id=raw, cache_dir=ms_cache)
            if local_dir and Path(local_dir).exists():
                return str(local_dir), "modelscope"
        except Exception:
            pass

    return raw, "huggingface_or_remote"
