from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short_name(name: str) -> str:
    return {
        "question_only": "Q only",
        "image_only": "Image only",
        "gold_evidence_text_only": "Gold text",
        "gold_evidence_image": "Gold text+image",
        "all_doc_text_only": "All text",
        "retrieved_text_only": "Retrieved text",
        "retrieved_image_only": "Retrieved image",
        "retrieved_evidence_image": "Retrieved text+image",
    }.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docqa", type=str, required=True)
    parser.add_argument("--chartqa", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    panels = [("DocQA", load_rows(Path(args.docqa)))]
    if args.chartqa:
        panels.append(("ChartQA", load_rows(Path(args.chartqa))))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.4), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, rows) in zip(axes, panels):
        labels = [short_name(r["baseline"]) for r in rows]
        scores = [float(r["score"]) for r in rows]
        colors = ["#2374ab" if "image" in r["baseline"] else "#f28e2b" for r in rows]
        ax.bar(range(len(rows)), scores, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, 0.7)
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Grounding Baselines: Does the VLM Need Image and Evidence?")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved grounding baseline figure to: {out}")


if __name__ == "__main__":
    main()
