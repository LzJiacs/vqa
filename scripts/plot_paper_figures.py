from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def _load_leaderboard(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _draw_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_axis_off()
    nodes = [
        ("Document / Chart", "image + question"),
        ("OCR + Layout", "regions, bbox, zone"),
        ("Dense Retriever", "BGE top-k evidence"),
        ("Cross Reranker", "query-aware ranking"),
        ("Answerability", "thresholded abstention"),
        ("Qwen2.5-VL", "evidence-grounded answer"),
        ("Evaluation", "score, evidence, refusal"),
    ]
    x0, y, w, h, gap = 0.03, 0.48, 0.12, 0.22, 0.025
    colors = ["#e9f5f2", "#edf1fb", "#fff2d8", "#fbe5e2", "#eef0f2", "#e8f4ff", "#f2eadf"]
    for i, (title, sub) in enumerate(nodes):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[i], edgecolor="#1b1b1b", linewidth=1.5))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.34, sub, ha="center", va="center", fontsize=8.5)
        if i < len(nodes) - 1:
            start = (x + w + 0.004, y + h / 2)
            end = (x + w + gap - 0.006, y + h / 2)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color="#1b1b1b"))
    ax.text(0.5, 0.2, "Evidence-driven and abstention-aware multimodal document QA", ha="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _draw_leaderboard(rows: list[dict], out: Path) -> None:
    labels = [r["trial"] for r in rows]
    doc = [float(r["docqa_vlm_score"]) for r in rows]
    chart = [float(r["chartqa_vlm_score"]) for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar([i - 0.18 for i in x], doc, width=0.36, label="DocQA", color="#2374ab")
    ax.bar([i + 0.18 for i in x], chart, width=0.36, label="ChartQA", color="#f28e2b")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Main Experiment Results")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _draw_reliability(summary_paths: list[Path], out: Path) -> None:
    labels: list[str] = []
    evidence: list[float] = []
    hallucination: list[float] = []
    abstain: list[float] = []
    for p in summary_paths:
        if not p.exists():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        labels.append(p.parent.name)
        evidence.append(float(obj.get("evidence_recall_at_rerank_k", 0.0)))
        hallucination.append(float(obj.get("hallucination_proxy_rate", 0.0)))
        abstain.append(float(obj.get("overall_abstain_rate", 0.0)))
    if not labels:
        return
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(x, evidence, marker="o", linewidth=2.2, label="Evidence recall@rerank-k", color="#2374ab")
    ax.plot(x, hallucination, marker="s", linewidth=2.2, label="Hallucination proxy", color="#c43c39")
    ax.plot(x, abstain, marker="^", linewidth=2.2, label="Abstain rate", color="#4c9f70")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Reliability Diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    exp = Path(args.experiment_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _draw_pipeline(out / "pipeline_overview.png")
    leaderboard = exp / "leaderboard.csv"
    if leaderboard.exists():
        _draw_leaderboard(_load_leaderboard(leaderboard), out / "main_results.png")

    report = exp / "report.json"
    reliability_paths: list[Path] = []
    if report.exists():
        obj = json.loads(report.read_text(encoding="utf-8"))
        for row in obj.get("all_trials", []):
            root = Path(row.get("results_root", ""))
            reliability_paths.append(root / "docqa_vlm_reliability.json")
    _draw_reliability(reliability_paths, out / "reliability_diagnostics.png")
    print(f"Saved paper figures to: {out}")


if __name__ == "__main__":
    main()
