from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import Prediction, QAItem


def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    args = parser.parse_args()

    preds = read_jsonl(args.pred, Prediction)
    gold = read_jsonl(args.gold, QAItem)

    pmap = {p.qid: p for p in preds}

    total = 0
    em = 0
    abstain_correct = 0
    abstain_total = 0

    for g in gold:
        if g.qid not in pmap:
            continue
        total += 1
        p = pmap[g.qid]

        if g.answerable:
            if normalize_text(p.predicted_answer) == normalize_text(g.answer):
                em += 1
        else:
            abstain_total += 1
            if p.abstain:
                abstain_correct += 1

    em_acc = em / max(total, 1)
    abstain_acc = abstain_correct / max(abstain_total, 1)
    score = np.mean([em_acc, abstain_acc])

    print(f"Total: {total}")
    print(f"EM: {em_acc:.4f}")
    print(f"AbstainAcc: {abstain_acc:.4f}")
    print(f"BalancedScore: {score:.4f}")


if __name__ == "__main__":
    main()
