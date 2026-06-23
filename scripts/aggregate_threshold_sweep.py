from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_eval(path: Path) -> dict:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    rows: list[dict] = []
    for rel_path in sorted(sweep.glob("thr_*_reliability.json")):
        m = re.search(r"thr_(\d+)p(\d+)_reliability", rel_path.name)
        if not m:
            continue
        threshold = float(f"{m.group(1)}.{m.group(2)}")
        tag = rel_path.name.replace("_reliability.json", "")
        eval_metrics = parse_eval(sweep / f"{tag}_eval.txt")
        rel = json.loads(rel_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "threshold": threshold,
                "score": eval_metrics.get("Score", 0.0),
                "task_accuracy_with_abstain": rel.get("task_accuracy_with_abstain", 0.0),
                "answerable_accuracy": rel.get("answerable_accuracy", 0.0),
                "answerable_abstain_rate": rel.get("answerable_abstain_rate", 0.0),
                "unanswerable_abstain_rate": rel.get("unanswerable_abstain_rate", 0.0),
                "hallucination_proxy_rate": rel.get("hallucination_proxy_rate", 0.0),
                "evidence_recall_at_rerank_k": rel.get("evidence_recall_at_rerank_k", 0.0),
            }
        )

    rows.sort(key=lambda x: x["threshold"])
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_prefix.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = [r["threshold"] for r in rows]
    ax.plot(x, [r["score"] for r in rows], marker="o", label="DocQA score", color="#2374ab", linewidth=2.2)
    ax.plot(
        x,
        [r["unanswerable_abstain_rate"] for r in rows],
        marker="s",
        label="Unanswerable abstain rate",
        color="#4c9f70",
        linewidth=2.2,
    )
    ax.plot(
        x,
        [r["hallucination_proxy_rate"] for r in rows],
        marker="^",
        label="Hallucination proxy",
        color="#c43c39",
        linewidth=2.2,
    )
    ax.set_xlabel("Answerability threshold")
    ax.set_ylabel("Rate / Score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Answerability Threshold Trade-off")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
