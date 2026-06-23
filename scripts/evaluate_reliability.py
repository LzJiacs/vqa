from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import Prediction, QAItem


REFUSAL = "i cannot answer from evidence"


def norm_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_number(text: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["docqa", "chartqa"], default="docqa")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    preds = read_jsonl(args.pred, Prediction)
    gold = read_jsonl(args.gold, QAItem)
    pmap = {p.qid: p for p in preds}

    total = correct = abstained = 0
    answerable_total = answerable_correct = answerable_abstain = 0
    unanswerable_total = unanswerable_abstain = unanswerable_answered = 0
    retrieval_hit = rerank_hit = 0
    evidence_total = 0
    refusal_text_count = 0

    for g in gold:
        p = pmap.get(g.qid)
        if p is None:
            continue
        total += 1
        pred_norm = norm_text(p.predicted_answer)
        is_refusal_text = REFUSAL in pred_norm
        if is_refusal_text:
            refusal_text_count += 1
        is_abstain = bool(p.abstain) or is_refusal_text
        abstained += int(is_abstain)

        if g.evidence_region_ids:
            evidence_total += 1
            gold_evidence = set(g.evidence_region_ids)
            retrieval_hit += int(bool(gold_evidence & set(p.retrieved_region_ids)))
            rerank_hit += int(bool(gold_evidence & set(p.reranked_region_ids)))

        if g.answerable:
            answerable_total += 1
            answerable_abstain += int(is_abstain)
            matched = relaxed_match(p.predicted_answer, g.answer, args.mode)
            correct += int(matched)
            answerable_correct += int(matched)
        else:
            unanswerable_total += 1
            unanswerable_abstain += int(is_abstain)
            unanswerable_answered += int(not is_abstain)
            correct += int(is_abstain)

    metrics = {
        "total": total,
        "task_accuracy_with_abstain": round(correct / max(total, 1), 6),
        "answerable_accuracy": round(answerable_correct / max(answerable_total, 1), 6),
        "answerable_abstain_rate": round(answerable_abstain / max(answerable_total, 1), 6),
        "unanswerable_abstain_rate": round(unanswerable_abstain / max(unanswerable_total, 1), 6),
        "hallucination_proxy_rate": round(unanswerable_answered / max(unanswerable_total, 1), 6),
        "overall_abstain_rate": round(abstained / max(total, 1), 6),
        "refusal_text_rate": round(refusal_text_count / max(total, 1), 6),
        "evidence_recall_at_retrieve_k": round(retrieval_hit / max(evidence_total, 1), 6),
        "evidence_recall_at_rerank_k": round(rerank_hit / max(evidence_total, 1), 6),
        "answerable_total": answerable_total,
        "unanswerable_total": unanswerable_total,
        "evidence_labeled_total": evidence_total,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
