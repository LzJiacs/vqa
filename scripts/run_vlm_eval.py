from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import read_jsonl, write_jsonl
from vqa4090.data.schemas import Prediction, QAItem, Region
from vqa4090.vlm.base import MockVLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--backend", type=str, default="qwen2.5-vl", choices=["mock", "qwen2.5-vl"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--image_limit", type=int, default=1)
    parser.add_argument("--vlm_max_evidence", type=int, default=5)
    parser.add_argument("--vlm_evidence_chars", type=int, default=320)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)
    if args.max_samples > 0:
        qas = qas[: args.max_samples]

    if args.backend == "mock":
        vlm = MockVLM()
    else:
        from vqa4090.vlm.qwen_client import QwenVLClient

        vlm = QwenVLClient(
            model_name=args.model,
            max_evidence=args.vlm_max_evidence,
            evidence_chars=args.vlm_evidence_chars,
        )

    by_doc: dict[str, list[Region]] = {}
    for r in regions:
        by_doc.setdefault(r.doc_id, []).append(r)

    preds: list[Prediction] = []
    for q in qas:
        regs = by_doc.get(q.doc_id, [])
        evidence = [r.text for r in regs[:3]]
        imgs = [r.image_path for r in regs if r.image_path][: max(args.image_limit, 1)]
        ans = vlm.answer(q.question, evidence, imgs)
        preds.append(
            Prediction(
                qid=q.qid,
                predicted_answer=ans,
                abstain=False,
                confidence=1.0,
                retrieved_region_ids=[r.region_id for r in regs[:3]],
                reranked_region_ids=[r.region_id for r in regs[:3]],
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.output, preds)
    print(f"Saved predictions: {args.output}")
    print(f"Samples: {len(preds)}")


if __name__ == "__main__":
    main()
