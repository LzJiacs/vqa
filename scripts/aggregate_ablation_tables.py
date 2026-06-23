from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_best(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def fmt(value: str) -> str:
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return value


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/project_profile_4090/20260417_121829")
    parser.add_argument("--output", type=str, default="docs/ablation_tables_emnlp.md")
    args = parser.parse_args()
    root = Path(args.root)

    specs = [
        ("global retrieval", root / "ablation_global_same_ans" / "leaderboard.csv"),
        ("doc-local retrieval", root / "inference_grid_doclocal_v3" / "leaderboard.csv"),
        ("no reranker", root / "ablation_noreranker_doclocal" / "leaderboard.csv"),
        ("strict support gate", root / "ablation_support_strict_repro" / "leaderboard.csv"),
        ("numeric support gate", root / "ablation_support_numeric_repro" / "leaderboard.csv"),
    ]
    rows = []
    for name, path in specs:
        row = read_best(path)
        rows.append(
            [
                name,
                row["score"],
                row["task_accuracy_with_abstain"],
                row["answerable_accuracy"],
                row["unanswerable_abstain_rate"],
                row["hallucination_proxy_rate"],
                row["evidence_recall_at_rerank_k"],
                row["overall_abstain_rate"],
                row["tag"],
            ]
        )

    text = "# Ablation Tables for EMNLP-Style Draft\n\n"
    text += markdown_table(
        [
            "variant",
            "score",
            "task_acc",
            "ans_acc",
            "unans_abstain",
            "halluc_proxy",
            "ev_recall_rerank",
            "abstain",
            "best_tag",
        ],
        rows,
    )
    text += "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "score",
                "task_acc",
                "ans_acc",
                "unans_abstain",
                "halluc_proxy",
                "ev_recall_rerank",
                "abstain",
                "best_tag",
            ]
        )
        writer.writerows(rows)
    print(f"Saved ablation table to: {output}")
    print(f"Saved ablation csv to: {csv_output}")


if __name__ == "__main__":
    main()
