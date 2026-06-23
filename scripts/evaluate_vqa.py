from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import Prediction, QAItem


def norm_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def extract_number(s: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    pn = extract_number(pred)
    gn = extract_number(gold)
    if pn is None or gn is None:
        return norm_text(pred) == norm_text(gold)
    if abs(gn) < 1e-8:
        return abs(pn - gn) <= 1e-6
    return abs(pn - gn) / abs(gn) <= tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--mode", type=str, default="docqa", choices=["docqa", "chartqa"])
    args = parser.parse_args()

    preds = read_jsonl(args.pred, Prediction)
    gold = read_jsonl(args.gold, QAItem)
    pmap = {p.qid: p for p in preds}

    total = 0
    exact = 0
    relaxed = 0
    for g in gold:
        p = pmap.get(g.qid)
        if p is None:
            continue
        total += 1
        if norm_text(p.predicted_answer) == norm_text(g.answer):
            exact += 1
        if args.mode == "chartqa":
            if relaxed_match(p.predicted_answer, g.answer):
                relaxed += 1
        else:
            if norm_text(p.predicted_answer) == norm_text(g.answer):
                relaxed += 1

    exact_acc = exact / max(total, 1)
    relaxed_acc = relaxed / max(total, 1)
    print(f"Total: {total}")
    print(f"ExactAcc: {exact_acc:.4f}")
    print(f"RelaxedAcc: {relaxed_acc:.4f}")
    print(f"Score: {np.mean([exact_acc, relaxed_acc]):.4f}")


if __name__ == "__main__":
    main()
