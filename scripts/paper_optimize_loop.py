from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_cmd(cmd: list[str], log_file: Path) -> tuple[int, str]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_file.open("w", encoding="utf-8", buffering=1) as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
            lines.append(line)
        code = proc.wait()
    return code, "".join(lines)


def parse_results_root(text: str) -> str | None:
    m = re.search(r"Results root:\s*([^\r\n]+)", text)
    return m.group(1).strip() if m else None


def write_profile(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def load_summary(summary_json: Path) -> dict:
    if not summary_json.exists():
        return {
            "docqa": {"score_gain": 0.0, "vlm": {"Score": 0.0}},
            "chartqa": {"score_gain": 0.0, "vlm": {"Score": 0.0}},
        }
    return json.loads(summary_json.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_exe", type=str, default=r"E:\anaconda\envs\vqa4090\python.exe")
    parser.add_argument("--workdir", type=str, default=r"D:\vqa")
    parser.add_argument("--doc_train", type=int, default=800)
    parser.add_argument("--doc_val", type=int, default=120)
    parser.add_argument("--doc_test", type=int, default=80)
    parser.add_argument("--chart_train", type=int, default=3000)
    parser.add_argument("--chart_val", type=int, default=400)
    parser.add_argument("--chart_test", type=int, default=80)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    exp_root = workdir / "experiments" / f"paper_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_root.mkdir(parents=True, exist_ok=True)
    profiles_dir = exp_root / "profiles"
    logs_dir = exp_root / "logs"
    records_csv = exp_root / "leaderboard.csv"
    records_json = exp_root / "leaderboard.json"

    trials = [
        {
            "name": "t1_base",
            "profile": {
                "docqa": {
                    "retriever_model": "BAAI/bge-base-en-v1.5",
                    "reranker_model": "BAAI/bge-reranker-base",
                    "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "retriever_epochs": 2,
                    "reranker_epochs": 2,
                    "batch_size": 24,
                    "retrieve_top_k": 12,
                    "rerank_top_k": 5,
                },
                "chartqa": {"vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct"},
            },
        },
        {
            "name": "t2_deeper_rerank",
            "profile": {
                "docqa": {
                    "retriever_model": "BAAI/bge-base-en-v1.5",
                    "reranker_model": "BAAI/bge-reranker-base",
                    "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "retriever_epochs": 3,
                    "reranker_epochs": 3,
                    "batch_size": 20,
                    "retrieve_top_k": 16,
                    "rerank_top_k": 8,
                },
                "chartqa": {"vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct"},
            },
        },
        {
            "name": "t3_large_retriever",
            "profile": {
                "docqa": {
                    "retriever_model": "BAAI/bge-large-en-v1.5",
                    "reranker_model": "BAAI/bge-reranker-base",
                    "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "retriever_epochs": 2,
                    "reranker_epochs": 2,
                    "batch_size": 16,
                    "retrieve_top_k": 16,
                    "rerank_top_k": 8,
                },
                "chartqa": {"vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct"},
            },
        },
        {
            "name": "t4_wide_recall",
            "profile": {
                "docqa": {
                    "retriever_model": "BAAI/bge-base-en-v1.5",
                    "reranker_model": "BAAI/bge-reranker-base",
                    "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "retriever_epochs": 2,
                    "reranker_epochs": 2,
                    "batch_size": 24,
                    "retrieve_top_k": 24,
                    "rerank_top_k": 10,
                },
                "chartqa": {"vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct"},
            },
        },
    ]

    rows: list[dict] = []
    for trial in trials:
        name = trial["name"]
        print(f"\n=== Start trial: {name} ===", flush=True)
        profile_path = profiles_dir / f"{name}.json"
        write_profile(profile_path, trial["profile"])

        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(workdir / "scripts" / "run_project_profile_4090.ps1"),
            "-PythonExe",
            args.python_exe,
            "-ProfileConfig",
            str(profile_path),
            "-DocTrain",
            str(args.doc_train),
            "-DocVal",
            str(args.doc_val),
            "-DocTest",
            str(args.doc_test),
            "-ChartTrain",
            str(args.chart_train),
            "-ChartVal",
            str(args.chart_val),
            "-ChartTest",
            str(args.chart_test),
        ]
        code, text = run_cmd(cmd, logs_dir / f"{name}.log")
        print(f"=== End trial: {name} (exit={code}) ===\n", flush=True)
        root = parse_results_root(text)
        summary = {}
        if code == 0 and root:
            summary = load_summary(Path(root) / "summary.json")
        else:
            summary = {
                "docqa": {"score_gain": 0.0, "vlm": {"Score": 0.0}},
                "chartqa": {"score_gain": 0.0, "vlm": {"Score": 0.0}},
            }

        row = {
            "trial": name,
            "status": "ok" if code == 0 else f"failed_{code}",
            "results_root": root or "",
            "docqa_vlm_score": summary["docqa"]["vlm"]["Score"],
            "docqa_gain": summary["docqa"]["score_gain"],
            "chartqa_vlm_score": summary["chartqa"]["vlm"]["Score"],
            "chartqa_gain": summary["chartqa"]["score_gain"],
            "combined_gain": round(float(summary["docqa"]["score_gain"]) + float(summary["chartqa"]["score_gain"]), 4),
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda x: x["combined_gain"], reverse=True)
    with records_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    records_json.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    best = rows_sorted[0]
    report = {
        "best_trial": best,
        "all_trials": rows_sorted,
        "notes": "All run logs and model checkpoints are preserved in each results_root.",
    }
    (exp_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    figures_dir = exp_root / "paper_figures"
    fig_cmd = [
        args.python_exe,
        str(workdir / "scripts" / "plot_paper_figures.py"),
        "--experiment_root",
        str(exp_root),
        "--output_dir",
        str(figures_dir),
    ]
    fig_code, _ = run_cmd(fig_cmd, logs_dir / "paper_figures.log")
    if fig_code != 0:
        print(f"Warning: paper figure generation failed with exit={fig_code}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Experiment root: {exp_root}", flush=True)


if __name__ == "__main__":
    main()
