from __future__ import annotations

import argparse
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
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--retriever", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--retrieve_top_k", type=int, default=8)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    qas = read_jsonl(args.qa, QAItem)
    regions = read_jsonl(args.regions, Region)

    encoder = DenseEncoder(args.retriever)
    reranker = CrossEncoderReranker(args.reranker)

    texts = [r.text for r in regions]
    region_emb = encoder.encode(texts, normalize=True)
    index = DenseIndex(region_emb.shape[1])
    index.add(region_emb)
    doc_to_indices: dict[str, list[int]] = {}
    for i, region in enumerate(regions):
        doc_to_indices.setdefault(region.doc_id, []).append(i)

    feats: list[AnswerabilityFeatures] = []
    labels: list[int] = []

    for q in qas:
        qvec = encoder.encode([q.question], normalize=True)
        if q.doc_id in doc_to_indices:
            candidate_indices = doc_to_indices[q.doc_id]
            sims = (qvec @ region_emb[candidate_indices].T)[0]
            order = np.argsort(-sims)[: args.retrieve_top_k].tolist()
            idx_list = [candidate_indices[i] for i in order]
            retrieved_scores = [float(sims[i]) for i in order]
        else:
            rs, idxs = index.search(qvec, top_k=args.retrieve_top_k)
            idx_list = idxs[0].tolist()
            retrieved_scores = [float(x) for x in rs[0].tolist()] if rs.size else []
        retrieved = [regions[i] for i in idx_list]
        rerank_scores = reranker.score(q.question, [r.text for r in retrieved])

        feats.append(
            build_answerability_features(
                question=q.question,
                retrieved_scores=retrieved_scores,
                rerank_scores=rerank_scores,
                evidence_lengths=[len(r.text) for r in retrieved],
            )
        )
        labels.append(int(q.answerable))

    model = AnswerabilityClassifier()
    model.fit(feats, labels)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "model.joblib"))
    print(f"Saved answerability model to: {out / 'model.joblib'}")


if __name__ == "__main__":
    main()
