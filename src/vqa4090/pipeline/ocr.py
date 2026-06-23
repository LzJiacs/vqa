from __future__ import annotations

from pathlib import Path

from PIL import Image


class SimpleOCR:
    """Placeholder OCR interface."""

    def run(self, image_path: str) -> list[dict]:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(image_path)

        Image.open(p).close()
        return []


class PaddleOCREngine:
    """PaddleOCR wrapper for document region extraction."""

    def __init__(self, lang: str = "en", use_gpu: bool = False, use_angle_cls: bool = True) -> None:
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise ImportError("PaddleOCR is not installed. Please install paddleocr and paddlepaddle.") from exc

        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )

    def run(self, image_path: str) -> list[dict]:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(image_path)

        Image.open(p).close()
        raw = self.ocr.ocr(str(p), cls=True)
        lines = raw[0] if raw and raw[0] else []
        items: list[dict] = []

        for i, line in enumerate(lines):
            if len(line) != 2:
                continue
            box, txt_score = line
            text = str(txt_score[0]).strip() if txt_score else ""
            score = float(txt_score[1]) if txt_score and len(txt_score) > 1 else 0.0
            if not text:
                continue

            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            items.append(
                {
                    "line_id": i,
                    "text": text,
                    "score": score,
                    "bbox": bbox,
                    "image_path": str(p),
                }
            )
        return items
