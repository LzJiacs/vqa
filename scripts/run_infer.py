from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.answerability.model import AnswerabilityClassifier
from vqa4090.data.io import read_jsonl, write_jsonl
from vqa4090.data.schemas import QAItem, Region
from vqa4090.pipeline.engine import EngineConfig, EvidenceVQAEngine
from vqa4090.rerank.cross_encoder import CrossEncoderReranker
from vqa4090.retrieval.encoder import DenseEncoder
from vqa4090.vlm.base import MockVLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--retriever", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--answerability_model", type=str, required=True)
    parser.add_argument("--abstain_threshold", type=float, default=0.4)
    parser.add_argument("--abstain_threshold_file", type=str, default="")
    parser.add_argument("--retrieve_top_k", type=int, default=8)
    parser.add_argument("--rerank_top_k", type=int, default=3)
    parser.add_argument("--vlm_backend", type=str, default="mock", choices=["mock", "qwen2.5-vl"])
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--vlm_max_evidence", type=int, default=5)
    parser.add_argument("--vlm_evidence_chars", type=int, default=320)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    abstain_threshold = args.abstain_threshold
    if args.abstain_threshold_file:
        p = Path(args.abstain_threshold_file)
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            abstain_threshold = float(obj.get("threshold", abstain_threshold))

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)

    encoder = DenseEncoder(args.retriever)
    reranker = CrossEncoderReranker(args.reranker)
    answerability = AnswerabilityClassifier.load(args.answerability_model)

    if args.vlm_backend == "mock":
        vlm = MockVLM()
    else:
        from vqa4090.vlm.qwen_client import QwenVLClient

        vlm = QwenVLClient(
            model_name=args.vlm_model,
            max_evidence=args.vlm_max_evidence,
            evidence_chars=args.vlm_evidence_chars,
        )

    engine = EvidenceVQAEngine(
        regions=regions,
        encoder=encoder,
        reranker=reranker,
        answerability=answerability,
        vlm=vlm,
        cfg=EngineConfig(
            retrieve_top_k=args.retrieve_top_k,
            rerank_top_k=args.rerank_top_k,
            abstain_threshold=abstain_threshold,
        ),
    )

    preds = engine.predict(qas)
    write_jsonl(args.output, preds)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
