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


def relaxed_match(pred: str, gold: str, mode: str = "docqa") -> bool:
    if mode != "chartqa":
        return norm_text(pred) == norm_text(gold)
    pn = extract_number(pred)
    gn = extract_number(gold)
    if pn is None or gn is None:
        return norm_text(pred) == norm_text(gold)
    if abs(gn) < 1e-8:
        return abs(pn - gn) <= 1e-6
    return abs(pn - gn) / abs(gn) <= 0.05


def evaluate(preds: list[Prediction], gold: list[QAItem]) -> dict:
    pmap = {p.qid: p for p in preds}
    total = exact = relaxed = 0
    task_correct = abstained = refusal_text_count = 0
    answerable_total = answerable_correct = answerable_abstain = 0
    unanswerable_total = unanswerable_abstain = unanswerable_answered = 0
    evidence_total = retrieval_hit = rerank_hit = 0

    for g in gold:
        p = pmap.get(g.qid)
        if p is None:
            continue
        total += 1
        pred_norm = norm_text(p.predicted_answer)
        is_refusal_text = REFUSAL in pred_norm
        is_abstain = bool(p.abstain) or is_refusal_text
        abstained += int(is_abstain)
        refusal_text_count += int(is_refusal_text)

        if norm_text(p.predicted_answer) == norm_text(g.answer):
            exact += 1
        if relaxed_match(p.predicted_answer, g.answer):
            relaxed += 1

        if g.evidence_region_ids:
            evidence_total += 1
            gold_evidence = set(g.evidence_region_ids)
            retrieval_hit += int(bool(gold_evidence & set(p.retrieved_region_ids)))
            rerank_hit += int(bool(gold_evidence & set(p.reranked_region_ids)))

        if g.answerable:
            answerable_total += 1
            answerable_abstain += int(is_abstain)
            matched = relaxed_match(p.predicted_answer, g.answer)
            answerable_correct += int(matched)
            task_correct += int(matched)
        else:
            unanswerable_total += 1
            unanswerable_abstain += int(is_abstain)
            unanswerable_answered += int(not is_abstain)
            task_correct += int(is_abstain)

    exact_acc = exact / max(total, 1)
    relaxed_acc = relaxed / max(total, 1)
    return {
        "total": total,
        "exact": round(exact_acc, 6),
        "relaxed": round(relaxed_acc, 6),
        "score": round((exact_acc + relaxed_acc) / 2.0, 6),
        "task_accuracy_with_abstain": round(task_correct / max(total, 1), 6),
        "answerable_accuracy": round(answerable_correct / max(answerable_total, 1), 6),
        "answerable_abstain_rate": round(answerable_abstain / max(answerable_total, 1), 6),
        "unanswerable_abstain_rate": round(unanswerable_abstain / max(unanswerable_total, 1), 6),
        "hallucination_proxy_rate": round(unanswerable_answered / max(unanswerable_total, 1), 6),
        "overall_abstain_rate": round(abstained / max(total, 1), 6),
        "refusal_text_rate": round(refusal_text_count / max(total, 1), 6),
        "evidence_recall_at_retrieve_k": round(retrieval_hit / max(evidence_total, 1), 6),
        "evidence_recall_at_rerank_k": round(rerank_hit / max(evidence_total, 1), 6),
    }


def parse_grid(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        pairs.append((int(left), int(right)))
    return pairs


def parse_evidence_grid(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        left, right = item.split("x")
        pairs.append((int(left), int(right)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--qa", type=str, default="data/docvqa/test/qa.jsonl")
    parser.add_argument("--regions", type=str, default="data/docvqa/test/regions.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--answerability_model", type=str, default="")
    parser.add_argument("--topk_grid", type=str, default="8:3,12:5,16:5,16:8,24:5,24:8")
    parser.add_argument("--thresholds", type=str, default="0.75")
    parser.add_argument("--evidence_grid", type=str, default="5x320")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--require_textual_support", action="store_true")
    parser.add_argument("--support_match_mode", type=str, default="strict", choices=["strict", "numeric"])
    parser.add_argument("--global_retrieval", action="store_true")
    parser.add_argument("--disable_reranker", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)
    encoder = DenseEncoder(str(root / "retriever"))
    reranker = CrossEncoderReranker(str(root / "reranker"))
    answerability_path = args.answerability_model or str(root / "answerability" / "model.joblib")
    answerability = AnswerabilityClassifier.load(answerability_path)
    vlm = QwenVLClient(model_name=args.vlm_model)

    rows: list[dict] = []
    topk_grid = parse_grid(args.topk_grid)
    evidence_grid = parse_evidence_grid(args.evidence_grid)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    for max_evidence, evidence_chars in evidence_grid:
        vlm.max_evidence = max_evidence
        vlm.evidence_chars = evidence_chars
        for retrieve_top_k, rerank_top_k in topk_grid:
            for threshold in thresholds:
                tag_parts = [
                    f"rt{retrieve_top_k}",
                    f"rk{rerank_top_k}",
                    f"ev{max_evidence}x{evidence_chars}",
                    f"th{threshold:.2f}",
                ]
                if args.global_retrieval:
                    tag_parts.append("global")
                if args.disable_reranker:
                    tag_parts.append("norerank")
                if args.require_textual_support:
                    tag_parts.append(f"support-{args.support_match_mode}")
                tag = "_".join(tag_parts).replace(".", "p")
                print(f"=== {tag} ===", flush=True)
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
                        doc_local_retrieval=not args.global_retrieval,
                        use_reranker=not args.disable_reranker,
                        require_textual_support=args.require_textual_support,
                        support_match_mode=args.support_match_mode,
                    ),
                )
                preds = engine.predict(qas)
                write_jsonl(pred_dir / f"{tag}.jsonl", preds)
                metrics = evaluate(preds, qas)
                row = {
                    "tag": tag,
                    "retrieve_top_k": retrieve_top_k,
                    "rerank_top_k": rerank_top_k,
                    "max_evidence": max_evidence,
                    "evidence_chars": evidence_chars,
                    "threshold": threshold,
                    **metrics,
                }
                rows.append(row)
                (out_dir / f"{tag}_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    rows.sort(key=lambda x: (x["task_accuracy_with_abstain"], x["score"], -x["hallucination_proxy_rate"]), reverse=True)
    with (out_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "leaderboard.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== leaderboard ===", flush=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
