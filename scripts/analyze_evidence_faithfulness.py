from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import Prediction, QAItem, Region

REFUSAL = "i cannot answer from evidence"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%$+-]+", " ", text)
    return " ".join(text.split())


def relaxed_contains(needle: str, haystack: str) -> bool:
    needle_n = normalize(needle)
    haystack_n = normalize(haystack)
    if not needle_n:
        return False
    if needle_n in haystack_n:
        return True
    if len(needle_n) <= 3:
        return re.search(rf"\b{re.escape(needle_n)}\b", haystack_n) is not None
    return False


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def is_abstain(pred: Prediction) -> bool:
    return bool(pred.abstain) or REFUSAL in normalize(pred.predicted_answer)


def evidence_text(pred: Prediction, region_map: dict[str, Region]) -> str:
    parts = [region_map[rid].text for rid in pred.reranked_region_ids if rid in region_map]
    return "\n".join(parts)


def summarize(preds: list[Prediction], qas: list[QAItem], regions: list[Region], label: str) -> dict:
    pred_map = {p.qid: p for p in preds}
    region_map = {r.region_id: r for r in regions}

    total = 0
    answerable = 0
    answered = 0
    abstained = 0
    correct = 0
    gold_supported = 0
    pred_supported = 0
    correct_and_gold_supported = 0
    correct_and_pred_supported = 0
    unsupported_correct = 0
    answered_without_gold_support = 0
    answered_without_pred_support = 0
    evidence_hit = 0
    evidence_total = 0

    for q in qas:
        pred = pred_map.get(q.qid)
        if pred is None:
            continue

        total += 1
        answerable += int(q.answerable)
        abst = is_abstain(pred)
        abstained += int(abst)
        answered += int(not abst)

        ev_text = evidence_text(pred, region_map)
        pred_ok = relaxed_contains(pred.predicted_answer, ev_text) if not abst else False
        gold_ok = relaxed_contains(q.answer, ev_text)
        match = exact_match(pred.predicted_answer, q.answer)

        correct += int(match)
        pred_supported += int(pred_ok)
        gold_supported += int(gold_ok)
        correct_and_gold_supported += int(match and gold_ok)
        correct_and_pred_supported += int(match and pred_ok)
        unsupported_correct += int(match and not gold_ok)
        answered_without_gold_support += int((not abst) and not gold_ok)
        answered_without_pred_support += int((not abst) and not pred_ok)

        if q.evidence_region_ids:
            evidence_total += 1
            evidence_hit += int(bool(set(q.evidence_region_ids) & set(pred.reranked_region_ids)))

    def rate(num: int, den: int) -> float:
        return round(num / max(den, 1), 6)

    return {
        "label": label,
        "total": total,
        "answerable": answerable,
        "answered": answered,
        "abstained": abstained,
        "exact": rate(correct, total),
        "evidence_recall_at_rerank": rate(evidence_hit, evidence_total),
        "gold_answer_in_evidence_rate": rate(gold_supported, total),
        "pred_answer_in_evidence_rate": rate(pred_supported, answered),
        "correct_and_gold_supported_rate": rate(correct_and_gold_supported, total),
        "correct_and_pred_supported_rate": rate(correct_and_pred_supported, total),
        "unsupported_correct_rate": rate(unsupported_correct, total),
        "answered_without_gold_support_rate": rate(answered_without_gold_support, answered),
        "answered_without_pred_support_rate": rate(answered_without_pred_support, answered),
        "abstain_rate": rate(abstained, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="LABEL=path/to/predictions.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for item in args.prediction:
        if "=" not in item:
            raise ValueError("--prediction must be LABEL=PATH")
        label, path = item.split("=", 1)
        preds = read_jsonl(path, Prediction)
        rows.append(summarize(preds, qas, regions, label))

    rows.sort(key=lambda x: (x["correct_and_gold_supported_rate"], x["exact"]), reverse=True)
    (out_dir / "faithfulness_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "faithfulness_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
