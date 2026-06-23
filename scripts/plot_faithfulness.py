from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    metrics = [
        "exact",
        "evidence_recall_at_rerank",
        "correct_and_gold_supported_rate",
        "answered_without_gold_support_rate",
        "abstain_rate",
    ]
    labels = {
        "exact": "Exact",
        "evidence_recall_at_rerank": "Evidence recall",
        "correct_and_gold_supported_rate": "Correct + supported",
        "answered_without_gold_support_rate": "Answered w/o gold support",
        "abstain_rate": "Abstain",
    }

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = range(len(df))
    width = 0.15
    offsets = [(-2 + i) * width for i in range(len(metrics))]
    colors = ["#28536B", "#C2948A", "#7EA16B", "#BC4B51", "#4C5B5C"]

    for metric, offset, color in zip(metrics, offsets, colors):
        ax.bar([i + offset for i in x], df[metric], width=width, label=labels[metric], color=color)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Evidence Faithfulness: Separating Accuracy from Verifiability")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
