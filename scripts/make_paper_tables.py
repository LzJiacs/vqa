from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/project_profile_4090/20260417_121829")
    parser.add_argument("--output", type=str, default="docs/paper_tables_emnlp.md")
    args = parser.parse_args()

    root = Path(args.root)
    sections: list[str] = ["# Paper Tables for EMNLP-Style Draft\n"]

    old_clean = read_json(root / "inference_grid_clean" / "leaderboard.json")[0]
    doclocal = read_json(root / "inference_grid_doclocal_v3" / "leaderboard.json")[0]
    support = read_json(root / "inference_grid_doclocal_supportgate" / "leaderboard.json")[0]
    numeric = read_json(root / "inference_grid_supportgate_numeric" / "leaderboard.json")[0]

    sections.append("## Main DocVQA Reliability Progression\n")
    sections.append(
        markdown_table(
            [
                "system",
                "score",
                "task_acc",
                "ans_acc",
                "unans_abstain",
                "halluc_proxy",
                "ev_recall_rerank",
                "abstain",
            ],
            [
                [
                    "global retrieval",
                    old_clean["score"],
                    old_clean["task_accuracy_with_abstain"],
                    old_clean["answerable_accuracy"],
                    old_clean["unanswerable_abstain_rate"],
                    old_clean["hallucination_proxy_rate"],
                    old_clean["evidence_recall_at_rerank_k"],
                    old_clean["overall_abstain_rate"],
                ],
                [
                    "doc-local",
                    doclocal["score"],
                    doclocal["task_accuracy_with_abstain"],
                    doclocal["answerable_accuracy"],
                    doclocal["unanswerable_abstain_rate"],
                    doclocal["hallucination_proxy_rate"],
                    doclocal["evidence_recall_at_rerank_k"],
                    doclocal["overall_abstain_rate"],
                ],
                [
                    "doc-local + support gate",
                    support["score"],
                    support["task_accuracy_with_abstain"],
                    support["answerable_accuracy"],
                    support["unanswerable_abstain_rate"],
                    support["hallucination_proxy_rate"],
                    support["evidence_recall_at_rerank_k"],
                    support["overall_abstain_rate"],
                ],
                [
                    "numeric support gate",
                    numeric["score"],
                    numeric["task_accuracy_with_abstain"],
                    numeric["answerable_accuracy"],
                    numeric["unanswerable_abstain_rate"],
                    numeric["hallucination_proxy_rate"],
                    numeric["evidence_recall_at_rerank_k"],
                    numeric["overall_abstain_rate"],
                ],
            ],
        )
    )

    faith = read_json(root / "faithfulness_supportgate" / "faithfulness_summary.json")
    keep = ["supportgate_best", "doclocal_v3_best", "retrieved_evidence_image", "image_only"]
    by_label = {r["label"]: r for r in faith}
    sections.append("\n## Faithfulness / Verifiability\n")
    sections.append(
        markdown_table(
            [
                "system",
                "exact",
                "correct_supported",
                "pred_supported",
                "unsupported_correct",
                "answered_without_support",
                "abstain",
            ],
            [
                [
                    label,
                    by_label[label]["exact"],
                    by_label[label]["correct_and_gold_supported_rate"],
                    by_label[label]["pred_answer_in_evidence_rate"],
                    by_label[label]["unsupported_correct_rate"],
                    by_label[label]["answered_without_pred_support_rate"],
                    by_label[label]["abstain_rate"],
                ]
                for label in keep
                if label in by_label
            ],
        )
    )

    docqa_bl = read_csv(root / "grounding_baselines_docqa_doclocal" / "leaderboard.csv")
    chartqa_bl = read_csv(root / "grounding_baselines_chartqa_doclocal" / "leaderboard.csv")
    sections.append("\n## DocQA Grounding Baselines\n")
    sections.append(
        markdown_table(
            ["baseline", "score", "answerable_acc", "ev_recall_rerank"],
            [[r["baseline"], r["score"], r["answerable_accuracy"], r["evidence_recall_at_rerank_k"]] for r in docqa_bl],
        )
    )
    sections.append("\n## ChartQA Grounding Baselines\n")
    sections.append(
        markdown_table(
            ["baseline", "score", "relaxed", "answerable_acc", "ev_recall_rerank"],
            [
                [r["baseline"], r["score"], r["relaxed"], r["answerable_accuracy"], r["evidence_recall_at_rerank_k"]]
                for r in chartqa_bl
            ],
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Saved paper tables to: {output}")


if __name__ == "__main__":
    main()
