from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class AnswerabilityFeatures:
    top_retrieval_score: float
    top_rerank_score: float
    question_len: int
    top_evidence_len: int
    mean_retrieval_score: float = 0.0
    std_retrieval_score: float = 0.0
    retrieval_margin: float = 0.0
    mean_rerank_score: float = 0.0
    std_rerank_score: float = 0.0
    rerank_margin: float = 0.0
    evidence_len_mean: float = 0.0
    evidence_len_max: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.top_retrieval_score,
            self.top_rerank_score,
            float(self.question_len),
            float(self.top_evidence_len),
            self.mean_retrieval_score,
            self.std_retrieval_score,
            self.retrieval_margin,
            self.mean_rerank_score,
            self.std_rerank_score,
            self.rerank_margin,
            self.evidence_len_mean,
            self.evidence_len_max,
        ], dtype=np.float32)


class AnswerabilityClassifier:
    def __init__(self) -> None:
        self.model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.06,
            l2_regularization=0.03,
            random_state=42,
        )

    def fit(self, feats: list[AnswerabilityFeatures], labels: list[int]) -> None:
        x = np.stack([f.to_array() for f in feats], axis=0)
        y = np.array(labels, dtype=np.int64)
        self.model.fit(x, y)

    def predict_proba(self, feat: AnswerabilityFeatures) -> float:
        x = feat.to_array().reshape(1, -1)
        n_features = getattr(self.model, "n_features_in_", x.shape[1])
        x = x[:, :n_features]
        return float(self.model.predict_proba(x)[0, 1])

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "version": 2}, path)

    @classmethod
    def load(cls, path: str) -> "AnswerabilityClassifier":
        obj = cls()
        loaded = joblib.load(path)
        obj.model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        return obj


def build_answerability_features(
    question: str,
    retrieved_scores: list[float],
    rerank_scores: list[float],
    evidence_lengths: list[int],
) -> AnswerabilityFeatures:
    retrieval = np.array(retrieved_scores, dtype=np.float32) if retrieved_scores else np.zeros(1, dtype=np.float32)
    rerank = np.array(rerank_scores, dtype=np.float32) if rerank_scores else np.zeros(1, dtype=np.float32)
    lengths = np.array(evidence_lengths, dtype=np.float32) if evidence_lengths else np.zeros(1, dtype=np.float32)
    retrieval_sorted = np.sort(retrieval)[::-1]
    rerank_sorted = np.sort(rerank)[::-1]
    retrieval_margin = float(retrieval_sorted[0] - retrieval_sorted[1]) if retrieval_sorted.size > 1 else 0.0
    rerank_margin = float(rerank_sorted[0] - rerank_sorted[1]) if rerank_sorted.size > 1 else 0.0
    top_evidence_len = int(lengths[int(np.argmax(rerank))]) if rerank_scores and evidence_lengths else 0
    return AnswerabilityFeatures(
        top_retrieval_score=float(retrieval_sorted[0]),
        top_rerank_score=float(rerank_sorted[0]),
        question_len=len(question),
        top_evidence_len=top_evidence_len,
        mean_retrieval_score=float(np.mean(retrieval)),
        std_retrieval_score=float(np.std(retrieval)),
        retrieval_margin=retrieval_margin,
        mean_rerank_score=float(np.mean(rerank)),
        std_rerank_score=float(np.std(rerank)),
        rerank_margin=rerank_margin,
        evidence_len_mean=float(np.mean(lengths)),
        evidence_len_max=float(np.max(lengths)),
    )
