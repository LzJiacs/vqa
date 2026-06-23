from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import write_jsonl
from vqa4090.data.schemas import QAItem, Region


def convert_split(ds_split, split_name: str, out_root: Path, max_samples: int) -> tuple[list[QAItem], list[Region]]:
    out_dir = out_root / split_name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    qas: list[QAItem] = []
    regions: list[Region] = []

    selected = ds_split.select(range(min(max_samples, len(ds_split))))
    for idx, row in enumerate(selected):
        doc_id = f"{split_name}_{idx:06d}"
        img: Image.Image = row["image"]
        rel_img = f"data/chartqa/{split_name}/images/{doc_id}.png"
        abs_img = img_dir / f"{doc_id}.png"
        img.save(abs_img)

        region_id = f"{doc_id}_r0"
        regions.append(
            Region(
                doc_id=doc_id,
                region_id=region_id,
                image_path=rel_img,
                text="Chart image containing bars/lines/labels and values.",
                bbox=[],
            )
        )

        answers = row.get("label", []) or []
        answer = str(answers[0]).strip() if answers else ""
        question = str(row.get("query", "")).strip()
        qas.append(
            QAItem(
                qid=f"{doc_id}_q0",
                doc_id=doc_id,
                question=question,
                answer=answer,
                answerable=bool(answer),
                evidence_region_ids=[region_id] if answer else [],
            )
        )

    write_jsonl(out_dir / "qa.jsonl", qas)
    write_jsonl(out_dir / "regions.jsonl", regions)
    return qas, regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="data/chartqa")
    parser.add_argument("--max_train", type=int, default=6000)
    parser.add_argument("--max_val", type=int, default=1000)
    parser.add_argument("--max_test", type=int, default=1000)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceM4/ChartQA")
    tr_qas, tr_regs = convert_split(ds["train"], "train", out_root, args.max_train)
    va_qas, va_regs = convert_split(ds["val"], "val", out_root, args.max_val)
    te_qas, te_regs = convert_split(ds["test"], "test", out_root, args.max_test)

    print(f"Saved train: qa={len(tr_qas)}, regions={len(tr_regs)}")
    print(f"Saved val  : qa={len(va_qas)}, regions={len(va_regs)}")
    print(f"Saved test : qa={len(te_qas)}, regions={len(te_regs)}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
