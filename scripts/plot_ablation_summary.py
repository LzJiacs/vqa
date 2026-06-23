from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with Path(args.table_csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels = [r["variant"] for r in rows]
    task = [float(r["task_acc"]) for r in rows]
    halluc = [float(r["halluc_proxy"]) for r in rows]
    recall = [float(r["ev_recall_rerank"]) for r in rows]

    x = range(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar([i - width for i in x], task, width=width, label="Task acc + abstain", color="#28536B")
    ax.bar(list(x), recall, width=width, label="Evidence recall@rerank", color="#7EA16B")
    ax.bar([i + width for i in x], halluc, width=width, label="Hallucination proxy", color="#BC4B51")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rate")
    ax.set_title("Ablation Summary: Evidence Boundary, Reranking, and Support Gate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"Saved ablation figure to: {output}")


if __name__ == "__main__":
    main()
