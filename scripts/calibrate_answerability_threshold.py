from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vqa4090.answerability.model import AnswerabilityClassifier, build_answerability_features
from vqa4090.data.io import read_jsonl
from vqa4090.data.schemas import QAItem, Region
from vqa4090.rerank.cross_encoder import CrossEncoderReranker
from vqa4090.retrieval.encoder import DenseEncoder
from vqa4090.retrieval.index import DenseIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True, help="Validation QA jsonl")
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--reranker", type=str, required=True)
    parser.add_argument("--answerability_model", type=str, required=True)
    parser.add_argument("--retrieve_top_k", type=int, default=8)
    parser.add_argument("--grid_min", type=float, default=0.1)
    parser.add_argument("--grid_max", type=float, default=0.9)
    parser.add_argument("--grid_step", type=float, default=0.02)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)

    encoder = DenseEncoder(args.retriever)
    reranker = CrossEncoderReranker(args.reranker)
    ans_model = AnswerabilityClassifier.load(args.answerability_model)

    texts = [r.text for r in regions]
    emb = encoder.encode(texts, normalize=True)
    index = DenseIndex(emb.shape[1])
    index.add(emb)
    doc_to_indices: dict[str, list[int]] = {}
    for i, region in enumerate(regions):
        doc_to_indices.setdefault(region.doc_id, []).append(i)

    probs: list[float] = []
    labels: list[int] = []
    for q in qas:
        qvec = encoder.encode([q.question], normalize=True)
        if q.doc_id in doc_to_indices:
            candidate_indices = doc_to_indices[q.doc_id]
            sims = (qvec @ emb[candidate_indices].T)[0]
            order = np.argsort(-sims)[: args.retrieve_top_k].tolist()
            idx_list = [candidate_indices[i] for i in order]
            retrieved_scores = [float(sims[i]) for i in order]
        else:
            rs, idxs = index.search(qvec, top_k=args.retrieve_top_k)
            idx_list = idxs[0].tolist()
            retrieved_scores = [float(x) for x in rs[0].tolist()] if rs.size else []
        retrieved = [regions[i] for i in idx_list]
        rerank_scores = reranker.score(q.question, [r.text for r in retrieved])

        feat = build_answerability_features(
            question=q.question,
            retrieved_scores=retrieved_scores,
            rerank_scores=rerank_scores,
            evidence_lengths=[len(r.text) for r in retrieved],
        )
        probs.append(ans_model.predict_proba(feat))
        labels.append(int(q.answerable))

    thresholds = np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step)
    best = {"threshold": 0.4, "balanced_acc": -1.0, "answerable_acc": 0.0, "unanswerable_acc": 0.0}
    y = np.array(labels, dtype=np.int64)
    p = np.array(probs, dtype=np.float32)

    for t in thresholds:
        pred_answerable = (p >= float(t)).astype(np.int64)
        pos_mask = y == 1
        neg_mask = y == 0
        ans_acc = float((pred_answerable[pos_mask] == 1).mean()) if pos_mask.any() else 0.0
        una_acc = float((pred_answerable[neg_mask] == 0).mean()) if neg_mask.any() else 0.0
        bal = (ans_acc + una_acc) / 2.0
        if bal > best["balanced_acc"]:
            best = {
                "threshold": float(round(float(t), 4)),
                "balanced_acc": float(round(bal, 6)),
                "answerable_acc": float(round(ans_acc, 6)),
                "unanswerable_acc": float(round(una_acc, 6)),
            }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False))
    print(f"Saved threshold file: {out}")


if __name__ == "__main__":
    main()
