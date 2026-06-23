from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from vqa4090.answerability.model import AnswerabilityClassifier, build_answerability_features
from vqa4090.data.schemas import Prediction, QAItem, Region
from vqa4090.rerank.cross_encoder import CrossEncoderReranker
from vqa4090.retrieval.encoder import DenseEncoder
from vqa4090.retrieval.index import DenseIndex
from vqa4090.vlm.base import VLMClient


@dataclass
class EngineConfig:
    retrieve_top_k: int = 8
    rerank_top_k: int = 3
    abstain_threshold: float = 0.4
    doc_local_retrieval: bool = True
    use_reranker: bool = True
    require_textual_support: bool = False
    support_match_mode: str = "strict"


def _normalize_for_support(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%$+-]+", " ", text)
    return " ".join(text.split())


def _extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", "")):
        try:
            numbers.append(float(match.group(0)))
        except ValueError:
            continue
    return numbers


def _numbers_match(answer: str, evidence: str) -> bool:
    answer_numbers = _extract_numbers(answer)
    if not answer_numbers:
        return False
    evidence_numbers = _extract_numbers(evidence)
    for answer_num in answer_numbers:
        for evidence_num in evidence_numbers:
            tolerance = max(1e-4, abs(answer_num) * 1e-4)
            if abs(answer_num - evidence_num) <= tolerance:
                return True
    return False


def _is_supported_by_evidence(answer: str, evidence_texts: list[str], match_mode: str = "strict") -> bool:
    evidence = "\n".join(evidence_texts)
    answer_norm = _normalize_for_support(answer)
    evidence_norm = _normalize_for_support(evidence)
    if not answer_norm or "i cannot answer from evidence" in answer_norm:
        return True
    if answer_norm in evidence_norm:
        return True
    if match_mode == "numeric" and _numbers_match(answer, evidence):
        return True
    if len(answer_norm) <= 3:
        return re.search(rf"\b{re.escape(answer_norm)}\b", evidence_norm) is not None
    return False


class EvidenceVQAEngine:
    def __init__(
        self,
        regions: list[Region],
        encoder: DenseEncoder,
        reranker: CrossEncoderReranker,
        answerability: AnswerabilityClassifier,
        vlm: VLMClient,
        cfg: EngineConfig | None = None,
    ) -> None:
        self.regions = regions
        self.encoder = encoder
        self.reranker = reranker
        self.answerability = answerability
        self.vlm = vlm
        self.cfg = cfg or EngineConfig()

        self.region_texts = [r.text for r in regions]
        self.region_ids = [r.region_id for r in regions]
        self.doc_to_indices: dict[str, list[int]] = {}
        for i, region in enumerate(regions):
            self.doc_to_indices.setdefault(region.doc_id, []).append(i)

        self.region_emb = self.encoder.encode(self.region_texts, normalize=True)
        self.index = DenseIndex(self.region_emb.shape[1])
        self.index.add(self.region_emb)

    def _retrieve(self, question: str, doc_id: str | None = None) -> tuple[list[str], list[float], list[Region]]:
        q = self.encoder.encode([question], normalize=True)
        if self.cfg.doc_local_retrieval and doc_id in self.doc_to_indices:
            candidate_indices = self.doc_to_indices[doc_id]
            candidate_emb = self.region_emb[candidate_indices]
            sims = (q @ candidate_emb.T)[0]
            order = np.argsort(-sims)[: self.cfg.retrieve_top_k].tolist()
            idx_list = [candidate_indices[i] for i in order]
            sc = [float(sims[i]) for i in order]
            ids = [self.region_ids[i] for i in idx_list]
            reg = [self.regions[i] for i in idx_list]
            return ids, sc, reg

        scores, idxs = self.index.search(q, top_k=self.cfg.retrieve_top_k)
        ids = [self.region_ids[i] for i in idxs[0].tolist()]
        sc = [float(x) for x in scores[0].tolist()]
        reg = [self.regions[i] for i in idxs[0].tolist()]
        return ids, sc, reg

    def _rerank(self, question: str, regs: list[Region]) -> tuple[list[Region], list[float]]:
        if not self.cfg.use_reranker:
            selected = regs[: self.cfg.rerank_top_k]
            return selected, [0.0 for _ in selected]
        docs = [r.text for r in regs]
        scores = self.reranker.score(question, docs)
        order = np.argsort(-np.array(scores))
        reranked = [regs[i] for i in order[: self.cfg.rerank_top_k].tolist()]
        rr_scores = [scores[i] for i in order[: self.cfg.rerank_top_k].tolist()]
        return reranked, rr_scores

    def predict_one(self, item: QAItem) -> Prediction:
        retrieved_ids, retrieved_scores, retrieved_regions = self._retrieve(item.question, item.doc_id)
        reranked_regions, rerank_scores = self._rerank(item.question, retrieved_regions)

        feat = build_answerability_features(
            question=item.question,
            retrieved_scores=retrieved_scores,
            rerank_scores=rerank_scores,
            evidence_lengths=[len(r.text) for r in reranked_regions],
        )
        p_answerable = self.answerability.predict_proba(feat)
        abstain = p_answerable < self.cfg.abstain_threshold

        if abstain:
            pred = "I cannot answer from evidence."
        else:
            evidence_texts = [x.text for x in reranked_regions]
            image_paths = [x.image_path for x in reranked_regions if x.image_path]
            pred = self.vlm.answer(item.question, evidence_texts, image_paths=image_paths)
            if self.cfg.require_textual_support and not _is_supported_by_evidence(
                pred, evidence_texts, self.cfg.support_match_mode
            ):
                abstain = True
                pred = "I cannot answer from evidence."

        return Prediction(
            qid=item.qid,
            predicted_answer=pred,
            abstain=abstain,
            confidence=p_answerable,
            retrieved_region_ids=retrieved_ids,
            reranked_region_ids=[x.region_id for x in reranked_regions],
        )

    def predict(self, items: list[QAItem]) -> list[Prediction]:
        return [self.predict_one(x) for x in items]
