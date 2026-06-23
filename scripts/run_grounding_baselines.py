from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.answerability.model import AnswerabilityClassifier
from vqa4090.data.io import read_jsonl, write_jsonl
from vqa4090.data.schemas import Prediction, QAItem, Region
from vqa4090.pipeline.engine import EngineConfig, EvidenceVQAEngine
from vqa4090.rerank.cross_encoder import CrossEncoderReranker
from vqa4090.retrieval.encoder import DenseEncoder
from vqa4090.vlm.qwen_client import QwenVLClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REFUSAL = "i cannot answer from evidence"


def norm_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_number(text: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def relaxed_match(pred: str, gold: str, mode: str) -> bool:
    if mode != "chartqa":
        return norm_text(pred) == norm_text(gold)
    pn = extract_number(pred)
    gn = extract_number(gold)
    if pn is None or gn is None:
        return norm_text(pred) == norm_text(gold)
    if abs(gn) < 1e-8:
        return abs(pn - gn) <= 1e-6
    return abs(pn - gn) / abs(gn) <= 0.05


def evaluate(preds: list[Prediction], gold: list[QAItem], mode: str) -> dict:
    pmap = {p.qid: p for p in preds}
    total = exact = relaxed = 0
    abstained = refusal_text_count = 0
    answerable_total = answerable_correct = 0
    evidence_total = retrieval_hit = rerank_hit = 0

    for g in gold:
        p = pmap.get(g.qid)
        if p is None:
            continue
        total += 1
        pred_norm = norm_text(p.predicted_answer)
        is_refusal = REFUSAL in pred_norm or bool(p.abstain)
        abstained += int(is_refusal)
        refusal_text_count += int(REFUSAL in pred_norm)

        exact += int(norm_text(p.predicted_answer) == norm_text(g.answer))
        relaxed += int(relaxed_match(p.predicted_answer, g.answer, mode))
        if g.answerable:
            answerable_total += 1
            answerable_correct += int(relaxed_match(p.predicted_answer, g.answer, mode))
        if g.evidence_region_ids:
            evidence_total += 1
            gold_evidence = set(g.evidence_region_ids)
            retrieval_hit += int(bool(gold_evidence & set(p.retrieved_region_ids)))
            rerank_hit += int(bool(gold_evidence & set(p.reranked_region_ids)))

    exact_acc = exact / max(total, 1)
    relaxed_acc = relaxed / max(total, 1)
    return {
        "total": total,
        "exact": round(exact_acc, 6),
        "relaxed": round(relaxed_acc, 6),
        "score": round((exact_acc + relaxed_acc) / 2.0, 6),
        "answerable_accuracy": round(answerable_correct / max(answerable_total, 1), 6),
        "overall_abstain_rate": round(abstained / max(total, 1), 6),
        "refusal_text_rate": round(refusal_text_count / max(total, 1), 6),
        "evidence_recall_at_retrieve_k": round(retrieval_hit / max(evidence_total, 1), 6),
        "evidence_recall_at_rerank_k": round(rerank_hit / max(evidence_total, 1), 6),
    }


def first_images(qas: list[QAItem], by_doc: dict[str, list[Region]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for q in qas:
        imgs = [r.image_path for r in by_doc.get(q.doc_id, []) if r.image_path]
        out[q.qid] = imgs[:1]
    return out


def make_direct_preds(
    qas: list[QAItem],
    by_doc: dict[str, list[Region]],
    region_map: dict[str, Region],
    vlm: QwenVLClient,
    baseline: str,
) -> list[Prediction]:
    preds: list[Prediction] = []
    qid_to_images = first_images(qas, by_doc)
    for q in qas:
        evidence: list[str] = []
        images: list[str] = []
        evidence_ids: list[str] = []
        if baseline == "question_only":
            pass
        elif baseline == "image_only":
            images = qid_to_images.get(q.qid, [])
        elif baseline == "gold_evidence_text_only":
            evidence_ids = q.evidence_region_ids
            evidence = [region_map[x].text for x in evidence_ids if x in region_map]
        elif baseline == "gold_evidence_image":
            evidence_ids = q.evidence_region_ids
            evidence = [region_map[x].text for x in evidence_ids if x in region_map]
            images = qid_to_images.get(q.qid, [])
        elif baseline == "all_doc_text_only":
            regs = by_doc.get(q.doc_id, [])[:8]
            evidence_ids = [r.region_id for r in regs]
            evidence = [r.text for r in regs]
        else:
            raise ValueError(f"Unknown direct baseline: {baseline}")

        answer = vlm.answer(q.question, evidence, image_paths=images)
        preds.append(
            Prediction(
                qid=q.qid,
                predicted_answer=answer,
                abstain=REFUSAL in norm_text(answer),
                confidence=1.0,
                retrieved_region_ids=evidence_ids,
                reranked_region_ids=evidence_ids,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return preds


def make_retrieved_preds(
    qas: list[QAItem],
    regions: list[Region],
    vlm: QwenVLClient,
    root: Path,
    baseline: str,
    retrieve_top_k: int,
    rerank_top_k: int,
    threshold: float,
) -> list[Prediction]:
    encoder = DenseEncoder(str(root / "retriever"))
    reranker = CrossEncoderReranker(str(root / "reranker"))
    answerability = AnswerabilityClassifier.load(str(root / "answerability" / "model.joblib"))
    engine = EvidenceVQAEngine(
        regions=regions,
        encoder=encoder,
        reranker=reranker,
        answerability=answerability,
        vlm=vlm,
        cfg=EngineConfig(
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            abstain_threshold=threshold,
        ),
    )
    preds: list[Prediction] = []
    for q in qas:
        retrieved_ids, _, retrieved_regions = engine._retrieve(q.question, q.doc_id)
        reranked_regions, _ = engine._rerank(q.question, retrieved_regions)
        evidence = [r.text for r in reranked_regions]
        images = [r.image_path for r in reranked_regions if r.image_path]
        if baseline == "retrieved_text_only":
            images = []
        elif baseline == "retrieved_image_only":
            evidence = []
        elif baseline == "retrieved_evidence_image":
            pass
        else:
            raise ValueError(f"Unknown retrieved baseline: {baseline}")

        answer = vlm.answer(q.question, evidence, image_paths=images[:1])
        preds.append(
            Prediction(
                qid=q.qid,
                predicted_answer=answer,
                abstain=REFUSAL in norm_text(answer),
                confidence=1.0,
                retrieved_region_ids=retrieved_ids,
                reranked_region_ids=[r.region_id for r in reranked_regions],
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["docqa", "chartqa"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--baselines", type=str, default="question_only,image_only,gold_evidence_text_only,gold_evidence_image")
    parser.add_argument("--retrieve_top_k", type=int, default=16)
    parser.add_argument("--rerank_top_k", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max_evidence", type=int, default=2)
    parser.add_argument("--evidence_chars", type=int, default=160)
    args = parser.parse_args()

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)
    by_doc: dict[str, list[Region]] = {}
    for r in regions:
        by_doc.setdefault(r.doc_id, []).append(r)
    region_map = {r.region_id: r for r in regions}

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    vlm = QwenVLClient(model_name=args.model, max_evidence=args.max_evidence, evidence_chars=args.evidence_chars)
    rows: list[dict] = []
    direct = {
        "question_only",
        "image_only",
        "gold_evidence_text_only",
        "gold_evidence_image",
        "all_doc_text_only",
    }
    retrieved = {"retrieved_text_only", "retrieved_image_only", "retrieved_evidence_image"}

    for baseline in [x.strip() for x in args.baselines.split(",") if x.strip()]:
        print(f"=== baseline: {baseline} ===", flush=True)
        if baseline in direct:
            preds = make_direct_preds(qas, by_doc, region_map, vlm, baseline)
        elif baseline in retrieved:
            if not args.root:
                raise ValueError(f"{baseline} requires --root")
            preds = make_retrieved_preds(
                qas,
                regions,
                vlm,
                Path(args.root),
                baseline,
                args.retrieve_top_k,
                args.rerank_top_k,
                args.threshold,
            )
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        pred_path = pred_dir / f"{baseline}.jsonl"
        write_jsonl(pred_path, preds)
        metrics = evaluate(preds, qas, args.mode)
        row = {
            "baseline": baseline,
            "mode": args.mode,
            "retrieve_top_k": args.retrieve_top_k if baseline in retrieved else "",
            "rerank_top_k": args.rerank_top_k if baseline in retrieved else "",
            "max_evidence": args.max_evidence,
            "evidence_chars": args.evidence_chars,
            **metrics,
        }
        rows.append(row)
        (out_dir / f"{baseline}_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    rows.sort(key=lambda x: x["score"], reverse=True)
    with (out_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "leaderboard.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
