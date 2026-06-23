from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import Prediction, QAItem, Region


def _resolve_image(path_text: str, workdir: Path) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return workdir / p


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    box = draw.textbbox(xy, text, font=font)
    pad = 3
    bg = (max(fill[0] - 40, 0), max(fill[1] - 40, 0), max(fill[2] - 40, 0))
    draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], fill=bg)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="D:/vqa")
    parser.add_argument("--max_images", type=int, default=12)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = read_jsonl(args.pred, Prediction)
    gold = read_jsonl(args.gold, QAItem)
    regions = read_jsonl(args.regions, Region)
    pmap = {p.qid: p for p in preds}
    qmap = {q.qid: q for q in gold}
    rmap = {r.region_id: r for r in regions}

    written = 0
    index_lines = ["qid\tanswer\tprediction\timage\n"]
    for qid, pred in pmap.items():
        if written >= args.max_images:
            break
        q = qmap.get(qid)
        if q is None:
            continue
        region_ids = pred.reranked_region_ids[:5] or pred.retrieved_region_ids[:5]
        selected = [rmap[x] for x in region_ids if x in rmap and rmap[x].image_path]
        if not selected:
            continue
        image_path = _resolve_image(selected[0].image_path or "", workdir)
        if not image_path.exists():
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = [(0, 150, 255), (255, 70, 70), (20, 170, 90), (255, 170, 0), (170, 70, 255)]
        for rank, region in enumerate(selected):
            if len(region.bbox) != 4:
                continue
            x0, y0, x1, y1 = region.bbox
            color = colors[rank % len(colors)]
            width = 5 if region.region_id in q.evidence_region_ids else 3
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            _draw_label(draw, (x0, max(0, y0 - 14)), f"R{rank + 1}", color)

        out_path = out_dir / f"{written:03d}_{qid}.png"
        img.save(out_path)
        index_lines.append(
            f"{qid}\t{q.answer}\t{pred.predicted_answer}\t{out_path.as_posix()}\n"
        )
        written += 1

    (out_dir / "index.tsv").write_text("".join(index_lines), encoding="utf-8")
    print(f"Saved {written} evidence visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
