# Related Work Notes for EMNLP-Style Revision

## Positioning

Our current project is best positioned as a verifiable and abstention-aware layer for multimodal document QA, rather than as a new end-to-end VLM. The nearest literature is multimodal document RAG, retrieval-augmented VQA, and evidence-grounded / self-critical RAG. The main novelty should be framed around:

1. document-scoped evidence boundary before VLM reasoning;
2. answerability-calibrated refusal;
3. evidence-verified abstention gate after VLM generation;
4. faithfulness-first evaluation that separates raw correctness from evidence-supported correctness.

## Closest Papers / Systems

| work | core idea | relation to our system | gap we target |
|---|---|---|---|
| SimpleDoc, EMNLP 2025 | Dual-cue page retrieval with iterative refinement for multimodal DocQA | Similar RAG-style document QA motivation; emphasizes retrieval quality and iterative missing-evidence repair | Does not make refusal / unsupported-correct auditing the central metric |
| MoLoRAG, EMNLP 2025 | Multi-modal logic-aware retrieval using page graph traversal | Supports our claim that semantic retrieval alone can be insufficient | We work at region/document evidence boundary and add answerability + support verification |
| CMRAG, arXiv 2025 | Co-modality document RAG using text and image channels | Supports our ChartQA finding that text-only is weak and text+image helps | We focus on verifiability and refusal, not only co-modal retrieval accuracy |
| RAVQA / RAVQA-v2 | Retrieval-augmented VQA with outside knowledge and fine-grained late interaction | General retrieval-augmented VQA baseline family | Targets outside knowledge VQA, not document-local evidence and abstention reliability |
| Self-RAG / active RAG family | Retrieve, generate, critique / self-reflection for RAG | Motivates evidence verification and critique after generation | Mostly text QA; our evidence is multimodal document regions |

## Method Improvements Inspired by Related Work

### 1. Document-Scoped Evidence Boundary

Inspired by the retrieval precision focus in SimpleDoc and MoLoRAG, we constrain candidate evidence to the question's own document before dense retrieval. This avoids cross-document contamination and turned out to be the largest current gain:

- evidence recall@rerank improved from about `0.5067` to `0.96`;
- hallucination proxy on unanswerable examples dropped from about `0.90` to `0.10`.

### 2. Evidence-Verified Abstention Gate

Inspired by self-critical RAG but implemented cheaply for local 4090 experiments, we add a post-generation support gate:

1. the VLM first answers using retrieved evidence and image;
2. if the generated answer cannot be matched in retrieved evidence text, the system refuses;
3. this converts offline faithfulness analysis into an online decision.

Best support-gate DocVQA setting:

| setting | value |
|---|---:|
| retrieve_top_k | 8 |
| rerank_top_k | 5 |
| max_evidence | 5 |
| evidence_chars | 320 |
| threshold | 0.08 |
| exact / relaxed / score | 0.55 / 0.55 / 0.55 |
| task accuracy with abstain | 0.73 |
| unanswerable abstain rate | 0.90 |
| hallucination proxy | 0.10 |
| pred answer in evidence rate | 1.00 |
| answered without predicted support | 0.00 |

### 3. Co-Modality Evidence Baseline

Inspired by CMRAG, we explicitly compare image-only, text-only, and retrieved text+image:

DocQA:

| system | score | evidence recall@rerank |
|---|---:|---:|
| retrieved text+image | 0.58 | 0.96 |
| image only | 0.57 | 0.00 |
| retrieved text only | 0.28 | 0.96 |

ChartQA:

| system | score | answerable accuracy |
|---|---:|---:|
| retrieved text+image | 0.60 | 0.65 |
| image only | 0.5625 | 0.60 |
| retrieved text only | 0.06875 | 0.10 |

Interpretation: the image channel is essential, but adding retrieved evidence improves ChartQA and makes DocQA answers auditable.

## Strongest Current Claim

Large VLMs can often answer document questions from the image alone, but raw correctness is not the same as verifiable correctness. Our system explicitly optimizes for evidence-supported answers and calibrated refusal. On the current DocQA subset, image-only reaches competitive raw accuracy but has `0.57` unsupported-correct rate, whereas our support-gated/doc-local evidence systems achieve `0.00` unsupported-correct rate with competitive exact accuracy and much stronger unanswerable handling.

## Remaining EMNLP Risks

1. Scale: current reported subset is still small; expand DocVQA/ChartQA evaluation if runtime allows.
2. Evidence matching proxy: textual support is strict and may under-count visually supported answers.
3. Novelty: doc-local retrieval alone is simple; the paper should present the full framework as evidence boundary + answerability + support-gated refusal + faithfulness metrics.
4. Baselines: add at least one external open-source retrieval VQA baseline or reproduce a simplified CMRAG-style two-channel fusion baseline if time allows.
