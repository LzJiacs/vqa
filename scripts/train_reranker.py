from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import QAItem, Region
from vqa4090.utils.model_resolver import resolve_model_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)

    examples: list[InputExample] = []
    for q in qas:
        positives = set(q.evidence_region_ids)
        for r in regions:
            if r.doc_id != q.doc_id:
                continue
            label = 1.0 if r.region_id in positives else 0.0
            examples.append(InputExample(texts=[q.question, r.text], label=label))

    resolved_model, provider = resolve_model_source(args.model)
    print(f"[train_reranker] model source: {provider} -> {resolved_model}")
    model = CrossEncoder(resolved_model, num_labels=1)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)
    model.fit(train_dataloader=loader, epochs=args.epochs, warmup_steps=0, show_progress_bar=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved reranker to: {out}")


if __name__ == "__main__":
    main()
