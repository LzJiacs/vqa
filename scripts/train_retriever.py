from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import QAItem, Region
from vqa4090.utils.model_resolver import resolve_model_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)
    region_map = {r.region_id: r for r in regions}

    train_examples: list[InputExample] = []
    for q in qas:
        if not q.answerable or not q.evidence_region_ids:
            continue
        pos = region_map[q.evidence_region_ids[0]].text
        train_examples.append(InputExample(texts=[q.question, pos]))

    resolved_model, provider = resolve_model_source(args.model)
    print(f"[train_retriever] model source: {provider} -> {resolved_model}")
    model = SentenceTransformer(resolved_model)
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, show_progress_bar=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved retriever to: {out}")


if __name__ == "__main__":
    main()
