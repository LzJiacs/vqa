from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboards", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows = load_rows([Path(x) for x in args.leaderboards])
    if not rows:
        raise SystemExit("No leaderboard rows found.")

    labels = [r["tag"] for r in rows]
    score = [float(r["score"]) for r in rows]
    evidence = [float(r["evidence_recall_at_rerank_k"]) for r in rows]
    hallucination = [float(r["hallucination_proxy_rate"]) for r in rows]
    x = range(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    axes[0].bar(x, score, color="#2374ab", label="DocQA score")
    axes[0].plot(x, evidence, color="#f28e2b", marker="o", linewidth=2.0, label="Evidence recall@rerank")
    axes[0].set_ylim(0, 0.65)
    axes[0].set_ylabel("Score / Recall")
    axes[0].set_title("Inference Grid: Accuracy and Evidence Recall")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x, hallucination, color="#c43c39", label="Hallucination proxy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Inference Grid: Reliability Cost")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved inference grid figure to: {out}")


if __name__ == "__main__":
    main()
