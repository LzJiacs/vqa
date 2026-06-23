from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


@dataclass
class RunConfig:
    name: str
    retriever_model: str
    reranker_model: str
    retriever_epochs: int
    reranker_epochs: int
    batch_size: int
    retrieve_top_k: int
    rerank_top_k: int
    abstain_threshold: float


def run_cmd(cmd: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")
    return output


def parse_eval_metrics(text: str) -> dict[str, float]:
    def find(name: str) -> float:
        m = re.search(rf"{name}:\s*([0-9]*\.?[0-9]+)", text)
        return float(m.group(1)) if m else 0.0

    return {
        "total": find("Total"),
        "em": find("EM"),
        "abstain_acc": find("AbstainAcc"),
        "balanced_score": find("BalancedScore"),
    }


def load_run_configs(path: Path) -> list[RunConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = []
    for row in data["runs"]:
        runs.append(RunConfig(**row))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--config", type=str, default="configs/sweep_docvqa.json")
    parser.add_argument("--train_qa", type=str, default="data/docvqa/train/qa.jsonl")
    parser.add_argument("--train_regions", type=str, default="data/docvqa/train/regions.jsonl")
    parser.add_argument("--test_qa", type=str, default="data/docvqa/test/qa.jsonl")
    parser.add_argument("--test_regions", type=str, default="data/docvqa/test/regions.jsonl")
    parser.add_argument("--root_out", type=str, default="outputs/sweeps")
    parser.add_argument("--vlm_backend", type=str, default="mock", choices=["mock", "qwen2.5-vl"])
    parser.add_argument("--seed_tag", type=str, default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fail_fast", action="store_true", default=False)
    args = parser.parse_args()

    runs = load_run_configs(Path(args.config))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.seed_tag}" if args.seed_tag else ""
    exp_dir = Path(args.root_out) / f"docvqa_sweep_{ts}{suffix}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = exp_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))

    summary_rows: list[dict[str, float | str]] = []

    global_idx = 0
    for rep in range(args.repeat):
        for run in runs:
            run_name = run.name if args.repeat == 1 else f"{run.name}_rep{rep + 1}"
            run_dir = exp_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(run.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

            retriever_out = run_dir / "retriever"
            reranker_out = run_dir / "reranker"
            answer_out = run_dir / "answerability"
            pred_out = run_dir / "predictions.jsonl"

            try:
                run_cmd(
                    [
                        args.python,
                        "scripts/train_retriever.py",
                        "--qa",
                        args.train_qa,
                        "--regions",
                        args.train_regions,
                        "--model",
                        run.retriever_model,
                        "--epochs",
                        str(run.retriever_epochs),
                        "--batch_size",
                        str(run.batch_size),
                        "--output",
                        str(retriever_out),
                    ],
                    run_dir / "train_retriever.log",
                )

                run_cmd(
                    [
                        args.python,
                        "scripts/train_reranker.py",
                        "--qa",
                        args.train_qa,
                        "--regions",
                        args.train_regions,
                        "--model",
                        run.reranker_model,
                        "--epochs",
                        str(run.reranker_epochs),
                        "--batch_size",
                        str(run.batch_size),
                        "--output",
                        str(reranker_out),
                    ],
                    run_dir / "train_reranker.log",
                )

                run_cmd(
                    [
                        args.python,
                        "scripts/train_answerability.py",
                        "--qa",
                        args.train_qa,
                        "--regions",
                        args.train_regions,
                        "--retriever",
                        str(retriever_out),
                        "--reranker",
                        str(reranker_out),
                        "--retrieve_top_k",
                        str(run.retrieve_top_k),
                        "--output",
                        str(answer_out),
                    ],
                    run_dir / "train_answerability.log",
                )

                infer_cmd = [
                    args.python,
                    "scripts/run_infer.py",
                    "--qa",
                    args.test_qa,
                    "--regions",
                    args.test_regions,
                    "--retriever",
                    str(retriever_out),
                    "--reranker",
                    str(reranker_out),
                    "--answerability_model",
                    str(answer_out / "model.joblib"),
                    "--abstain_threshold",
                    str(run.abstain_threshold),
                    "--retrieve_top_k",
                    str(run.retrieve_top_k),
                    "--rerank_top_k",
                    str(run.rerank_top_k),
                    "--vlm_backend",
                    args.vlm_backend,
                    "--output",
                    str(pred_out),
                ]
                run_cmd(infer_cmd, run_dir / "run_infer.log")

                eval_out = run_cmd(
                    [
                        args.python,
                        "scripts/evaluate.py",
                        "--pred",
                        str(pred_out),
                        "--gold",
                        args.test_qa,
                    ],
                    run_dir / "evaluate.log",
                )
                metrics = parse_eval_metrics(eval_out)
                status = "ok"
            except Exception as exc:
                metrics = {"total": 0.0, "em": 0.0, "abstain_acc": 0.0, "balanced_score": 0.0}
                status = f"failed: {exc}"
                (run_dir / "failed.txt").write_text(str(exc), encoding="utf-8")
                if args.fail_fast:
                    raise

            row = {
                "run_name": run_name,
                "status": status,
                "retriever_model": run.retriever_model,
                "reranker_model": run.reranker_model,
                "retriever_epochs": run.retriever_epochs,
                "reranker_epochs": run.reranker_epochs,
                "batch_size": run.batch_size,
                "retrieve_top_k": run.retrieve_top_k,
                "rerank_top_k": run.rerank_top_k,
                "abstain_threshold": run.abstain_threshold,
                **metrics,
            }
            summary_rows.append(row)

            writer.add_scalar("eval/em", metrics["em"], global_idx)
            writer.add_scalar("eval/abstain_acc", metrics["abstain_acc"], global_idx)
            writer.add_scalar("eval/balanced_score", metrics["balanced_score"], global_idx)
            writer.add_hparams(
                {
                    "retrieve_top_k": run.retrieve_top_k,
                    "rerank_top_k": run.rerank_top_k,
                    "abstain_threshold": run.abstain_threshold,
                    "batch_size": run.batch_size,
                    "retriever_epochs": run.retriever_epochs,
                    "reranker_epochs": run.reranker_epochs,
                },
                {
                    "hparam/em": metrics["em"],
                    "hparam/abstain_acc": metrics["abstain_acc"],
                    "hparam/balanced_score": metrics["balanced_score"],
                },
                run_name=run_name,
            )

            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            global_idx += 1

    summary_csv = exp_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(summary_rows)

    (exp_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    writer.close()
    print(f"Sweep completed: {exp_dir}")
    print(f"TensorBoard logdir: {tb_dir}")


if __name__ == "__main__":
    main()
