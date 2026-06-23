from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.leaderboard))
    points = []
    for row in rows:
        coverage = 1.0 - float(row["overall_abstain_rate"])
        risk = float(row["hallucination_proxy_rate"])
        task_acc = float(row["task_accuracy_with_abstain"])
        score = float(row["score"])
        points.append((coverage, risk, task_acc, score, row["tag"]))
    points.sort(key=lambda x: x[0])

    fig, ax1 = plt.subplots(figsize=(7.8, 5.2))
    coverage = [p[0] for p in points]
    risk = [p[1] for p in points]
    task_acc = [p[2] for p in points]
    score = [p[3] for p in points]

    ax1.plot(coverage, risk, color="#BC4B51", marker="o", label="Unanswerable hallucination proxy")
    ax1.plot(coverage, task_acc, color="#28536B", marker="s", label="Task accuracy with abstain")
    ax1.plot(coverage, score, color="#7EA16B", marker="^", label="DocQA score")
    ax1.set_xlabel("Coverage (1 - abstain rate)")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25)
    ax1.set_title("Risk-Coverage Trade-off Under Evidence-Gated Refusal")
    ax1.legend(frameon=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    print(f"Saved risk-coverage figure to: {output}")


if __name__ == "__main__":
    main()
