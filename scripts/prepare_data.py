from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import write_jsonl
from vqa4090.data.schemas import QAItem, Region


def make_sample_images(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    img1 = Image.new("RGB", (1000, 700), color=(250, 250, 250))
    d1 = ImageDraw.Draw(img1)
    d1.text((40, 40), "Invoice #A-1008", fill=(0, 0, 0))
    d1.text((40, 120), "Total Amount: $256.40", fill=(0, 0, 0))
    d1.text((40, 200), "Date: 2026-01-12", fill=(0, 0, 0))
    img1.save(root / "doc_invoice.png")

    img2 = Image.new("RGB", (1000, 700), color=(248, 248, 255))
    d2 = ImageDraw.Draw(img2)
    d2.text((40, 40), "Medical Report", fill=(0, 0, 0))
    d2.text((40, 120), "Patient: Alice Brown", fill=(0, 0, 0))
    d2.text((40, 200), "Diagnosis: Hypertension", fill=(0, 0, 0))
    img2.save(root / "doc_medical.png")


def make_sample_data(data_root: Path) -> None:
    regions = [
        Region(doc_id="doc1", region_id="doc1_r0", image_path="data/sample/images/doc_invoice.png", text="Invoice number is A-1008", bbox=[40, 40, 400, 80]),
        Region(doc_id="doc1", region_id="doc1_r1", image_path="data/sample/images/doc_invoice.png", text="Total Amount: $256.40", bbox=[40, 120, 500, 160]),
        Region(doc_id="doc1", region_id="doc1_r2", image_path="data/sample/images/doc_invoice.png", text="Date: 2026-01-12", bbox=[40, 200, 450, 240]),
        Region(doc_id="doc2", region_id="doc2_r0", image_path="data/sample/images/doc_medical.png", text="Medical report for patient Alice Brown", bbox=[40, 120, 600, 170]),
        Region(doc_id="doc2", region_id="doc2_r1", image_path="data/sample/images/doc_medical.png", text="Diagnosis: Hypertension", bbox=[40, 200, 500, 250]),
    ]

    qas = [
        QAItem(
            qid="q1",
            doc_id="doc1",
            question="What is the total amount in the invoice?",
            answer="$256.40",
            answerable=True,
            evidence_region_ids=["doc1_r1"],
        ),
        QAItem(
            qid="q2",
            doc_id="doc2",
            question="What diagnosis is listed in the medical report?",
            answer="Hypertension",
            answerable=True,
            evidence_region_ids=["doc2_r1"],
        ),
        QAItem(
            qid="q3",
            doc_id="doc1",
            question="What is the customer's phone number?",
            answer="",
            answerable=False,
            evidence_region_ids=[],
        ),
    ]

    write_jsonl(data_root / "regions.jsonl", regions)
    write_jsonl(data_root / "qa.jsonl", qas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", default=True)
    parser.add_argument("--out", type=str, default="data/sample")
    args = parser.parse_args()

    out = Path(args.out)
    make_sample_images(out / "images")
    make_sample_data(out)
    print(f"Sample data generated at: {out}")


if __name__ == "__main__":
    main()
