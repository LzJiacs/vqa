from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run(cmd: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8", buffering=1) as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        code = proc.wait()
    if code != 0:
        raise SystemExit(code)


def parse_eval(path: Path) -> dict:
    out: dict[str, float] = {}
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
    parser.add_argument("--python_exe", type=str, default=r"E:\anaconda\envs\vqa4090\python.exe")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--qa", type=str, default="data/docvqa/test/qa.jsonl")
    parser.add_argument("--regions", type=str, default="data/docvqa/test/regions.jsonl")
    parser.add_argument("--thresholds", type=str, default="0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--retrieve_top_k", type=int, default=12)
    parser.add_argument("--rerank_top_k", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / "threshold_sweep"
    logs = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for threshold in [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]:
        tag = f"thr_{threshold:.2f}".replace(".", "p")
        pred = out_dir / f"{tag}_pred.jsonl"
        eval_txt = out_dir / f"{tag}_eval.txt"
        reliability = out_dir / f"{tag}_reliability.json"

        run(
            [
                args.python_exe,
                "scripts/run_infer.py",
                "--qa",
                args.qa,
                "--regions",
                args.regions,
                "--retriever",
                str(root / "retriever"),
                "--reranker",
                str(root / "reranker"),
                "--answerability_model",
                str(root / "answerability" / "model.joblib"),
                "--abstain_threshold",
                str(threshold),
                "--retrieve_top_k",
                str(args.retrieve_top_k),
                "--rerank_top_k",
                str(args.rerank_top_k),
                "--vlm_backend",
                "qwen2.5-vl",
                "--vlm_model",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "--output",
                str(pred),
            ],
            logs / f"{tag}_infer.log",
        )
        run(
            [args.python_exe, "scripts/evaluate_vqa.py", "--pred", str(pred), "--gold", args.qa, "--mode", "docqa"],
            logs / f"{tag}_eval.log",
        )
        eval_txt.write_text((logs / f"{tag}_eval.log").read_text(encoding="utf-8"), encoding="utf-8")
        run(
            [
                args.python_exe,
                "scripts/evaluate_reliability.py",
                "--pred",
                str(pred),
                "--gold",
                args.qa,
                "--mode",
                "docqa",
                "--output",
                str(reliability),
            ],
            logs / f"{tag}_reliability.log",
        )
        eval_metrics = parse_eval(eval_txt)
        rel = json.loads(reliability.read_text(encoding="utf-8"))
        rows.append(
            {
                "threshold": threshold,
                "score": eval_metrics.get("Score", 0.0),
                "exact": eval_metrics.get("ExactAcc", 0.0),
                "task_accuracy_with_abstain": rel.get("task_accuracy_with_abstain", 0.0),
                "answerable_accuracy": rel.get("answerable_accuracy", 0.0),
                "answerable_abstain_rate": rel.get("answerable_abstain_rate", 0.0),
                "unanswerable_abstain_rate": rel.get("unanswerable_abstain_rate", 0.0),
                "hallucination_proxy_rate": rel.get("hallucination_proxy_rate", 0.0),
                "evidence_recall_at_rerank_k": rel.get("evidence_recall_at_rerank_k", 0.0),
            }
        )

    rows.sort(key=lambda x: (x["task_accuracy_with_abstain"], x["score"]), reverse=True)
    with (out_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "leaderboard.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
