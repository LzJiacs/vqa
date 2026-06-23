from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import write_jsonl
from vqa4090.pipeline.ocr import PaddleOCREngine, SimpleOCR
from vqa4090.pipeline.regions import build_regions_from_ocr


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ocr_backend", type=str, default="paddle", choices=["paddle", "simple"])
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--merge_lines", action="store_true", default=True)
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(docs_dir)

    if args.ocr_backend == "paddle":
        ocr = PaddleOCREngine(lang=args.lang, use_gpu=args.use_gpu)
    else:
        ocr = SimpleOCR()

    images = sorted([p for p in docs_dir.rglob("*") if p.is_file() and is_image_file(p)])
    regions_all: list[dict] = []

    for img_path in images:
        doc_id = img_path.stem
        ocr_items = ocr.run(str(img_path))
        with Image.open(img_path) as img:
            page_size = img.size
        regions = build_regions_from_ocr(
            doc_id=doc_id,
            ocr_items=ocr_items,
            merge_lines=args.merge_lines,
            page_size=page_size,
        )
        for r in regions:
            regions_all.append(r.model_dump())

    write_jsonl(args.output, regions_all)
    print(f"Images: {len(images)}")
    print(f"Regions: {len(regions_all)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
