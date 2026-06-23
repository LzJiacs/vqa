from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import write_jsonl
from vqa4090.data.schemas import QAItem, Region


def _choose_question(query: dict) -> str:
    for lang in ("en", "de", "fr", "es", "it"):
        text = str(query.get(lang, "")).strip()
        if text:
            return text
    return ""


def _choose_answer(sample: dict) -> str:
    answer_obj = sample.get("answer", {}) or {}
    text = str(answer_obj.get("text", "")).strip()
    if text:
        return text
    answers = sample.get("answers", []) or []
    for a in answers:
        t = str(a).strip()
        if t:
            return t
    return ""


def _safe_bbox(box: list[float]) -> list[int]:
    if len(box) != 4:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = box
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def _merge_bbox(boxes: list[list[float]]) -> list[int]:
    if not boxes:
        return [0, 0, 0, 0]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return _safe_bbox([x0, y0, x1, y1])


def _score_overlap(answer: str, chunk_text: str) -> float:
    a = set(answer.lower().split())
    c = set(chunk_text.lower().split())
    if not a or not c:
        return 0.0
    inter = len(a & c)
    return inter / max(len(a), 1)


def _zone_from_bbox(bbox: list[int], page_w: int, page_h: int) -> str:
    if len(bbox) != 4 or page_w <= 0 or page_h <= 0:
        return "unknown"
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    horiz = "left" if cx < page_w * 0.5 else "right"
    if cy < page_h * 0.18:
        vert = "header"
    elif cy > page_h * 0.86:
        vert = "footer"
    else:
        vert = "body"
    return f"{vert}_{horiz}"


def _convert_split(
    samples,
    split_name: str,
    out_root: Path,
    chunk_words: int,
    max_samples: int,
    rng: random.Random,
    neg_ratio: float,
    add_layout_tokens: bool,
) -> tuple[list[QAItem], list[Region]]:
    split_dir = out_root / split_name
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    regions: list[Region] = []
    qas: list[QAItem] = []

    selected = samples.select(range(min(max_samples, len(samples))))

    for row in selected:
        doc_id = f"{split_name}_{row['id']}"
        img: Image.Image = row["image"]
        page_w, page_h = img.size
        img_rel = f"data/docvqa/{split_name}/images/{doc_id}.png"
        img_path = image_dir / f"{doc_id}.png"
        img.save(img_path)

        words = [str(w).strip() for w in (row.get("words", []) or []) if str(w).strip()]
        bboxes = row.get("bounding_boxes", []) or []
        if not words:
            continue

        region_ids: list[str] = []
        region_texts: list[str] = []
        for i in range(0, len(words), chunk_words):
            j = min(i + chunk_words, len(words))
            chunk = words[i:j]
            chunk_boxes = bboxes[i:j] if i < len(bboxes) else []
            region_id = f"{doc_id}_r{i // chunk_words}"
            text = " ".join(chunk)
            bbox = _merge_bbox(chunk_boxes)
            if add_layout_tokens:
                zone = _zone_from_bbox(bbox, page_w, page_h)
                if zone != "unknown":
                    text = f"[{zone}] {text}"
            regions.append(
                Region(
                    doc_id=doc_id,
                    region_id=region_id,
                    image_path=img_rel,
                    text=text,
                    bbox=bbox,
                )
            )
            region_ids.append(region_id)
            region_texts.append(text)

        question = _choose_question(row.get("query", {}) or {})
        answer = _choose_answer(row)
        answerable = bool(answer.strip())

        evidence_ids: list[str] = []
        if answerable and region_texts:
            scores = [_score_overlap(answer, t) for t in region_texts]
            best = max(range(len(scores)), key=lambda x: scores[x])
            if scores[best] > 0:
                evidence_ids = [region_ids[best]]

        qas.append(
            QAItem(
                qid=f"{doc_id}_q0",
                doc_id=doc_id,
                question=question,
                answer=answer,
                answerable=answerable,
                evidence_region_ids=evidence_ids,
            )
        )

    # Build synthetic unanswerable samples by cross-document question swap.
    positive = [q for q in qas if q.answerable]
    doc_to_q = {q.doc_id: q for q in positive}
    doc_ids = list(doc_to_q.keys())
    neg_target = int(len(positive) * neg_ratio)
    neg_count = 0

    while doc_ids and neg_count < neg_target:
        doc_id = rng.choice(doc_ids)
        src_doc = rng.choice(doc_ids)
        if src_doc == doc_id:
            continue
        src_q = doc_to_q[src_doc]
        qas.append(
            QAItem(
                qid=f"{doc_id}_neg{neg_count}",
                doc_id=doc_id,
                question=src_q.question,
                answer="",
                answerable=False,
                evidence_region_ids=[],
            )
        )
        neg_count += 1

    write_jsonl(split_dir / "regions.jsonl", regions)
    write_jsonl(split_dir / "qa.jsonl", qas)
    return qas, regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="data/docvqa")
    parser.add_argument("--chunk_words", type=int, default=24)
    parser.add_argument("--max_train", type=int, default=1000)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=200)
    parser.add_argument("--negative_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_layout_tokens", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    ds = load_dataset("nielsr/docvqa_1200_examples")
    train_end = min(args.max_train, len(ds["train"]))
    val_end = min(train_end + args.max_val, len(ds["train"]))
    train_split = ds["train"].select(range(train_end))
    val_split = ds["train"].select(range(train_end, val_end)) if args.max_val > 0 and val_end > train_end else None

    train_qas, train_regions = _convert_split(
        train_split,
        "train",
        out_root,
        args.chunk_words,
        args.max_train,
        rng,
        args.negative_ratio,
        not args.no_layout_tokens,
    )
    if val_split is not None:
        val_qas, val_regions = _convert_split(
            val_split,
            "val",
            out_root,
            args.chunk_words,
            len(val_split),
            rng,
            args.negative_ratio,
            not args.no_layout_tokens,
        )
    else:
        val_qas, val_regions = [], []
    test_qas, test_regions = _convert_split(
        ds["test"],
        "test",
        out_root,
        args.chunk_words,
        args.max_test,
        rng,
        args.negative_ratio,
        not args.no_layout_tokens,
    )

    print(f"Saved train: qa={len(train_qas)}, regions={len(train_regions)}")
    if val_split is not None:
        print(f"Saved val  : qa={len(val_qas)}, regions={len(val_regions)}")
    print(f"Saved test : qa={len(test_qas)}, regions={len(test_regions)}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
