from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    kv = {}
    for k in ["Total", "ExactAcc", "RelaxedAcc", "Score"]:
        m = re.search(rf"{k}:\s*([0-9]*\.?[0-9]+)", text)
        kv[k] = float(m.group(1)) if m else 0.0
    return kv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    doc_mock = parse_metrics(root / "docqa_mock_eval.txt")
    doc_vlm = parse_metrics(root / "docqa_vlm_eval.txt")
    chart_mock = parse_metrics(root / "chartqa_mock_eval.txt")
    chart_vlm = parse_metrics(root / "chartqa_vlm_eval.txt")

    summary = {
        "docqa": {
            "baseline_mock": doc_mock,
            "vlm": doc_vlm,
            "score_gain": round(doc_vlm["Score"] - doc_mock["Score"], 4),
        },
        "chartqa": {
            "baseline_mock": chart_mock,
            "vlm": chart_vlm,
            "score_gain": round(chart_vlm["Score"] - chart_mock["Score"], 4),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
