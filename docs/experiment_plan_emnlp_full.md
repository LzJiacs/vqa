# Full EMNLP-Oriented Experiment Plan

## Goal

Build a paper-ready experimental package for VERA: Verifiable Evidence-Routed Abstention for multimodal document QA. The central question is not only whether a VLM answers correctly, but whether the answer is supported by retrieved document evidence and whether the system refuses when evidence is insufficient.

## Research Questions

1. Does document-scoped evidence retrieval reduce cross-document evidence contamination?
2. Does reranking improve evidence recall and downstream supported accuracy?
3. Does evidence-aware VLM prompting improve over image-only and text-only baselines?
4. Does answerability calibration reduce unanswerable hallucination?
5. Does support-gated refusal improve verifiability without excessive coverage loss?
6. Does the method transfer from DocVQA-style documents to ChartQA-style visual reasoning?

## Core Metrics

| metric | purpose |
|---|---|
| exact / relaxed score | conventional task quality |
| task accuracy with abstain | answerable correct plus unanswerable refusal |
| answerable accuracy | answer quality when evidence exists |
| unanswerable abstain rate | refusal quality |
| hallucination proxy | unanswerable examples that were answered |
| evidence recall@retrieve / rerank | evidence localization |
| correct + supported | verifiable correctness |
| unsupported correct | VLM got answer right but not supported by retrieved evidence |
| answered without support | online faithfulness failure |
| coverage | 1 - abstain rate |

## Experiment Matrix

### A. Retrieval Boundary Ablation

| variant | description | expected role |
|---|---|---|
| global retrieval | search all regions in split | negative control; exposes cross-document contamination |
| doc-local retrieval | search only regions from current document | main method |

### B. Reranking Ablation

| variant | description |
|---|---|
| no reranker | dense retrieval order only |
| cross-encoder reranker | BGE reranker on retrieved candidates |

### C. Reasoning Context Ablation

| variant | description |
|---|---|
| question only | no image, no evidence |
| image only | direct VLM |
| retrieved text only | OCR/layout evidence only |
| retrieved image only | image without evidence text |
| retrieved text + image | co-modal evidence-aware reasoning |
| gold evidence + image | oracle evidence upper bound |

### D. Refusal / Verification Ablation

| variant | description |
|---|---|
| no support gate | answerability threshold only |
| strict support gate | predicted answer must textually appear in retrieved evidence |
| numeric support gate | strict plus numeric equivalence |

### E. Hyperparameter Sweeps

| parameter | values |
|---|---|
| retrieve_top_k | 8, 12, 16, 24 |
| rerank_top_k | 5, 8 |
| max_evidence | 2, 3, 5 |
| evidence_chars | 160, 240, 320 |
| abstain_threshold | 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.40 |

### F. Dataset Coverage

| dataset | current role | next target |
|---|---|---|
| DocVQA subset | main reliability and refusal setting | expand test subset beyond 100 if runtime permits |
| ChartQA subset | co-modality transfer | keep grounding baselines and add support-gate only if unanswerable labels exist |

## Current Completed Experiments

1. global retrieval baseline;
2. doc-local retrieval main run;
3. strict support-gated run;
4. numeric support-gated run;
5. DocQA grounding baselines;
6. ChartQA grounding baselines;
7. faithfulness analysis;
8. risk-coverage visualization.

## Experiments To Run Next

1. reranker ablation under doc-local retrieval;
2. global vs doc-local with identical answerability model and threshold;
3. strict vs numeric support gate with identical top-k and threshold;
4. expanded DocVQA evaluation if more prepared samples are available;
5. optional Qwen2.5-VL-7B scale ablation if ModelScope download succeeds.

## Paper Tables / Figures

| artifact | file |
|---|---|
| main results table | `docs/paper_tables_emnlp.md` |
| experiment report | `docs/experiment_report_20260417.md` |
| related work notes | `docs/related_work_emnlp_notes.md` |
| risk-coverage figure | `outputs/project_profile_4090/20260417_121829/figures/risk_coverage_supportgate.png` |
| grounding baseline figure | `outputs/project_profile_4090/20260417_121829/figures/grounding_baselines_doclocal_docqa_chartqa.png` |
