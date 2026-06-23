# Evidence-Driven VQA Experiment Report

Date: 2026-04-17

## Current Best Main Result

The best practical configuration is `profile_next_evidence_dev` with `chunk_words=24`, layout-aware region tokens, expanded Qwen2.5-VL evidence context, and a tuned answerability threshold of `0.75`.

| Setting | DocQA Score | ChartQA Score | Task Acc with Abstain | Unanswerable Abstain | Hallucination Proxy | Evidence Recall@Rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original t1 baseline | 0.43 | 0.625 | 0.45 | 0.10 | 0.90 | 0.3867 |
| Layout + expanded evidence, dev threshold 0.80 | 0.24 | 0.60 | 0.37 | 0.65 | 0.35 | 0.36 |
| Layout + expanded evidence, tuned threshold 0.75 | 0.48 | 0.60 | 0.51 | 0.15 | 0.85 | 0.36 |
| Inference grid best: retrieve 16 / rerank 8 / evidence 2x160 / threshold 0.75 | 0.50 | 0.60 | 0.52 | 0.10 | 0.90 | 0.3867 |
| Evidence-recall best: retrieve 24 / rerank 8 / evidence 3x240 / threshold 0.75 | 0.50 | 0.60 | 0.52 | 0.10 | 0.90 | 0.5067 |
| Safety mode: retrieve 16 / rerank 8 / evidence 2x160 / threshold 0.80 | 0.25 | 0.60 | 0.37 | 0.60 | 0.40 | 0.3867 |
| Chunk 16, tuned threshold 0.35-0.75 | 0.43 | 0.60 | 0.45 | 0.10 | 0.90 | 0.3867 |

## Interpretation

The expanded evidence prompt and inference-side top-k tuning improve DocQA answer accuracy from `0.43` to `0.50`. The best evidence-recall configuration reaches reranked evidence recall `0.5067`, but hallucination remains high at the accuracy-oriented threshold. The dev-calibrated threshold is conservative and strongly reduces hallucination proxy rate, but it over-abstains on answerable samples. This is a useful reliability trade-off rather than a pure accuracy win.

For a paper narrative, the most defensible framing is:

1. Accuracy-oriented mode: threshold `0.75`, DocQA `0.50`, ChartQA `0.60`, best task accuracy with abstain.
2. Safety-oriented mode: threshold `0.80`, lower DocQA score but hallucination proxy drops from `0.90` to about `0.40`.
3. Region granularity ablation: chunk `16` improves retrieve-level recall but does not improve final answer accuracy, suggesting reranker/VLM evidence use is now the bottleneck.

## Grounding Baselines

These baselines test whether the VLM can answer without explicit evidence and whether our evidence pipeline improves traceability.

| Baseline | DocQA Score | ChartQA Score | Interpretation |
| --- | ---: | ---: | --- |
| Question only | 0.00 | 0.0313 | Language prior alone is not enough. |
| Image only | 0.57 | 0.5625 | Qwen2.5-VL can answer many examples directly from the image. |
| Retrieved text only | 0.21 | N/A | Retrieved OCR evidence alone is insufficient. |
| Retrieved image only | 0.46 | N/A | Cropped/document image signal dominates text snippets. |
| Retrieved text + image | 0.50 | 0.60 | Our auditable pipeline is competitive but not yet better than direct image-only on DocQA. |
| Gold evidence text only | 0.34 | N/A | Gold evidence text helps but cannot replace visual context. |
| Gold evidence text + image | 0.58 | N/A | Oracle evidence gives an upper bound; retrieval remains a bottleneck. |

The grounding baseline changes the paper framing: the system should not be pitched as a new end-to-end model that simply beats Qwen2.5-VL. The stronger and more honest claim is a verifiable evidence layer around VLMs: it provides retrieved/cited evidence, abstention control, threshold trade-off curves, and evidence visualization. The experimental target should be calibrated reliability and auditability, not only raw VQA accuracy.

## Generated Artifacts

Main old sweep:

- `experiments/paper_opt_20260417_003533/leaderboard.csv`
- `experiments/paper_opt_20260417_003533/paper_figures/pipeline_overview.png`
- `experiments/paper_opt_20260417_003533/paper_figures/main_results.png`

Improved chunk-24 run:

- `outputs/project_profile_4090/20260417_121829/summary.json`
- `outputs/project_profile_4090/20260417_121829/docqa_vlm_reliability.json`
- `outputs/project_profile_4090/20260417_121829/chartqa_vlm_reliability.json`
- `outputs/project_profile_4090/20260417_121829/evidence_vis/docqa/`
- `outputs/project_profile_4090/20260417_121829/threshold_sweep/threshold_tradeoff.csv`
- `outputs/project_profile_4090/20260417_121829/threshold_sweep/threshold_tradeoff.png`
- `outputs/project_profile_4090/20260417_121829/inference_grid_clean/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/inference_grid_clean_safety/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/inference_grid_clean/inference_grid_summary.png`
- `outputs/project_profile_4090/20260417_121829/grounding_baselines_docqa/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/grounding_baselines_chartqa/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/grounding_baselines.png`

Chunk-size ablation:

- `outputs/project_profile_4090/20260417_123401/summary.json`
- `outputs/project_profile_4090/20260417_123401/docqa_vlm_reliability.json`
- `outputs/project_profile_4090/20260417_123401/threshold_sweep/threshold_tradeoff.csv`
- `outputs/project_profile_4090/20260417_123401/threshold_sweep/threshold_tradeoff.png`

## Next Ablations

The next experiments should isolate where the remaining bottleneck comes from:

1. Train a stronger answerability model with hard negatives and features from multiple reranked candidates, because the current threshold curve has a sharp cliff between `0.75` and `0.80`.
2. Improve reranking objective with harder in-document negatives; evidence recall can reach `0.5067`, but accuracy saturates at `0.50`.
3. Compare 3B vs 7B Qwen2.5-VL on the same cached predictions pipeline if GPU memory allows.
4. Region source: dataset OCR boxes vs PaddleOCR-extracted regions on a small diagnostic subset.
5. Add evidence citation scoring, not only answer exact match.

## Current Paper Claim Boundary

The current system is not an end-to-end foundation model. It is a verifiable evidence-and-abstention layer for multimodal document QA. The paper artifact should be a framework plus reproducible benchmark suite: retrieval/reranking modules, Qwen2.5-VL evidence-aware inference, abstention calibration, grounding baselines, evidence visualization, and reliability metrics. It is not yet a full EMNLP/NeurIPS-level result, but it has a clearer research story after the grounding baselines: direct VLMs can answer from images, while our system aims to make those answers auditable and rejectable.

## 2026-04-17 Doc-Local Evidence Update

### Method Change

The previous retriever searched over all regions in the evaluation split, which allowed evidence from unrelated documents to enter the prompt. I changed the retrieval boundary to doc-local retrieval: for each QA item, dense retrieval is constrained to regions whose `doc_id` matches the question document. This is a method-level correction and also a useful paper ablation, because it formalizes a document-grounded evidence boundary before VLM reasoning.

### Best DocVQA Result After Doc-Local Retrieval

| setting | value |
|---|---:|
| model | Qwen2.5-VL-3B-Instruct |
| retriever | BGE base, doc-local |
| reranker | BGE reranker |
| retrieve_top_k | 12 |
| rerank_top_k | 8 |
| max_evidence | 5 |
| evidence_chars | 320 |
| abstain_threshold | 0.24 |
| exact / relaxed / score | 0.54 / 0.54 / 0.54 |
| task accuracy with abstain | 0.72 |
| answerable accuracy | 0.675 |
| unanswerable abstain rate | 0.90 |
| hallucination proxy | 0.10 |
| evidence recall@retrieve | 0.9867 |
| evidence recall@rerank | 0.96 |

Compared with the earlier clean system, doc-local retrieval improves evidence recall@rerank from about 0.5067 to 0.96 and reduces the unanswerable hallucination proxy from about 0.90 to 0.10. The main reliability result is therefore substantially stronger than the earlier global-retrieval run.

### Faithfulness / Verifiability Result

| system | exact | correct + gold-supported | unsupported correct | answered without gold support | abstain |
|---|---:|---:|---:|---:|---:|
| doclocal_v3_refusal | 0.55 | 0.55 | 0.00 | 0.1169 | 0.23 |
| retrieved_evidence_image | 0.59 | 0.59 | 0.00 | 0.1724 | 0.13 |
| retrieved_image_only | 0.57 | 0.54 | 0.03 | 0.2174 | 0.08 |
| image_only | 0.57 | 0.00 | 0.57 | 1.00 | 0.08 |

This directly supports the claim that a VLM can answer many DocVQA questions from the image alone, but image-only correctness is not verifiable by retrieved evidence. The doc-local evidence system trades a small amount of raw accuracy for refusal and evidence support: unsupported correct answers drop to 0.00 in the measured subset, while image-only has 0.57 unsupported correct rate by construction.

### Doc-Local Grounding Baselines

| baseline | score | answerable accuracy | evidence recall@rerank |
|---|---:|---:|---:|
| retrieved_evidence_image | 0.58 | 0.725 | 0.96 |
| gold_evidence_image | 0.58 | 0.725 | 1.00 |
| retrieved_image_only | 0.57 | 0.7125 | 0.96 |
| image_only | 0.57 | 0.7125 | 0.00 |
| retrieved_text_only | 0.28 | 0.35 | 0.96 |
| question_only | 0.00 | 0.00 | 0.00 |

The retrieved evidence+image baseline now matches gold evidence+image on score, which indicates that doc-local retrieval plus reranking is close to oracle evidence for this subset. The gap between retrieved_text_only and retrieved_evidence_image also shows that the visual channel is still essential; the contribution is not to replace the image, but to force visual answers through a verifiable evidence boundary and refusal mechanism.

### New Artifacts

- `outputs/project_profile_4090/20260417_121829/inference_grid_doclocal_v3/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/inference_grid_doclocal_v3/predictions/`
- `outputs/project_profile_4090/20260417_121829/answerability_doclocal_v3/model.joblib`
- `outputs/project_profile_4090/20260417_121829/faithfulness_doclocal_v3/faithfulness_summary.csv`
- `outputs/project_profile_4090/20260417_121829/faithfulness_doclocal_baselines/faithfulness_summary.csv`
- `outputs/project_profile_4090/20260417_121829/grounding_baselines_docqa_doclocal/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/figures/inference_grid_doclocal_v3.png`
- `outputs/project_profile_4090/20260417_121829/figures/faithfulness_doclocal_v3.png`
- `outputs/project_profile_4090/20260417_121829/figures/faithfulness_doclocal_baselines.png`
- `outputs/project_profile_4090/20260417_121829/figures/grounding_baselines_doclocal.png`

### 7B Status

I attempted to download `Qwen/Qwen2.5-VL-7B-Instruct` from ModelScope. The download process ran for about one hour but did not create a usable local 7B cache directory, so I stopped the stalled process. The current validated results are still based on Qwen2.5-VL-3B-Instruct. A later 7B run should be treated as a model-scale ablation rather than the main method contribution.

## 2026-04-17 EMNLP-Oriented Additions

### Evidence-Verified Abstention Gate

I added an optional post-generation support gate. When enabled, the VLM first answers with retrieved evidence and image; then the system checks whether the predicted answer is textually supported by the retrieved evidence. If the predicted answer is unsupported, the system converts the output into a refusal. This makes faithfulness analysis part of inference rather than only an offline diagnostic.

Best support-gated DocVQA setting:

| setting | value |
|---|---:|
| retrieve_top_k | 8 |
| rerank_top_k | 5 |
| max_evidence | 5 |
| evidence_chars | 320 |
| abstain_threshold | 0.08 |
| exact / relaxed / score | 0.55 / 0.55 / 0.55 |
| task accuracy with abstain | 0.73 |
| answerable accuracy | 0.6875 |
| unanswerable abstain rate | 0.90 |
| hallucination proxy | 0.10 |
| abstain rate | 0.30 |

Faithfulness comparison:

| system | exact | correct + supported | pred answer in evidence | answered without pred support | unsupported correct |
|---|---:|---:|---:|---:|---:|
| supportgate_best | 0.56 | 0.56 | 1.00 | 0.00 | 0.00 |
| doclocal_v3_best | 0.55 | 0.55 | 0.9221 | 0.0779 | 0.00 |
| image_only | 0.57 | 0.00 | 0.00 | 1.00 | 0.57 |

The support gate slightly improves task accuracy (`0.73` vs `0.72`) and gives the cleanest verifiability story: every answered prediction is text-supported by retrieved evidence under the current proxy.

### ChartQA Doc-Local Grounding Baseline

| baseline | score | relaxed | answerable accuracy | evidence recall@rerank |
|---|---:|---:|---:|---:|
| retrieved_evidence_image | 0.60 | 0.65 | 0.65 | 1.00 |
| retrieved_image_only | 0.5625 | 0.60 | 0.60 | 1.00 |
| image_only | 0.5625 | 0.60 | 0.60 | 0.00 |
| retrieved_text_only | 0.06875 | 0.10 | 0.10 | 1.00 |
| question_only | 0.03125 | 0.0625 | 0.0625 | 0.00 |

ChartQA supports the co-modality claim: image-only is strong, text-only is weak, and retrieved evidence+image improves over image-only. This makes the paper less DocVQA-specific.

### Related Work Direction

The nearest work includes SimpleDoc-style multimodal DocQA retrieval, MoLoRAG-style logic-aware retrieval, CMRAG-style co-modal retrieval-augmented generation, and RAVQA-style retrieval-augmented VQA. The differentiator for this project should be stated as verifiability and abstention, not just retrieval-augmented accuracy.

New notes:

- `docs/related_work_emnlp_notes.md`
- `outputs/project_profile_4090/20260417_121829/inference_grid_doclocal_supportgate/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/faithfulness_supportgate/faithfulness_summary.csv`
- `outputs/project_profile_4090/20260417_121829/grounding_baselines_chartqa_doclocal/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/figures/inference_grid_doclocal_supportgate.png`
- `outputs/project_profile_4090/20260417_121829/figures/grounding_baselines_doclocal_docqa_chartqa.png`

## 2026-04-20 Optimization Update

### Numeric-Aware Support Gate

I extended the support gate with numeric matching for integers, decimals, percentages, and currency-like answers. This is intended to avoid over-refusing when the evidence contains the same number in a slightly different surface form, such as `975`, `$975.00`, or `975.0`.

The numeric-aware variant is useful as an ablation but did not replace the strict support gate as the main setting:

| system | score | task acc | answerable acc | unanswerable abstain | hallucination proxy | abstain |
|---|---:|---:|---:|---:|---:|---:|
| strict support gate | 0.55 | 0.73 | 0.6875 | 0.90 | 0.10 | 0.30 |
| numeric support gate | 0.55 | 0.72 | 0.6875 | 0.85 | 0.15 | 0.23 |

Interpretation: numeric matching reduces some refusals, but it also allows more unanswerable examples through. For the EMNLP-style reliability story, the stricter support gate remains cleaner because it maximizes verified answering and keeps hallucination proxy lower.

### Paper-Ready Tables and Risk-Coverage Figure

I added an automatic table generation script to prevent manual result transcription errors. It currently produces:

- main DocVQA reliability progression;
- faithfulness / verifiability comparison;
- DocQA grounding baseline table;
- ChartQA grounding baseline table.

I also added a risk-coverage plot for the support-gated system. This figure visualizes the trade-off between answer coverage, task accuracy, and unanswerable hallucination proxy under different thresholds.

New artifacts:

- `docs/paper_tables_emnlp.md`
- `scripts/make_paper_tables.py`
- `scripts/plot_risk_coverage.py`
- `outputs/project_profile_4090/20260417_121829/inference_grid_supportgate_numeric/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/figures/inference_grid_supportgate_numeric.png`
- `outputs/project_profile_4090/20260417_121829/figures/risk_coverage_supportgate.png`

## 2026-04-20 Full Experiment Matrix Update

I added a full EMNLP-style experiment plan and made the core pipeline switches explicit:

- `doc_local_retrieval`: compare global retrieval and document-scoped retrieval.
- `use_reranker`: compare dense retrieval only and cross-encoder reranking.
- `support_match_mode`: compare strict textual support and numeric-aware support.

Key ablation results:

| variant | score | task acc | answerable acc | unanswerable abstain | hallucination proxy | evidence recall@rerank | abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| global retrieval | 0.48 | 0.50 | 0.60 | 0.10 | 0.90 | 0.36 | 0.07 |
| doc-local retrieval | 0.54 | 0.72 | 0.675 | 0.90 | 0.10 | 0.96 | 0.23 |
| no reranker | 0.39 | 0.57 | 0.4875 | 0.90 | 0.10 | 0.96 | 0.42 |
| strict support gate | 0.55 | 0.73 | 0.6875 | 0.90 | 0.10 | 0.9067 | 0.30 |
| numeric support gate | 0.55 | 0.72 | 0.6875 | 0.85 | 0.15 | 0.9067 | 0.25 |

Interpretation:

1. Document-scoped retrieval is the largest reliability improvement: global retrieval has low evidence recall and high hallucination proxy.
2. Reranking is necessary for answer quality: no-reranker retains high evidence recall but drops score and answerable accuracy substantially.
3. Strict support gate gives the cleanest verifiability result, while numeric support gate increases coverage at the cost of more unsupported answers.

Faithfulness reproduction:

| system | exact | predicted answer supported | answered without predicted support | abstain |
|---|---:|---:|---:|---:|
| strict_repro | 0.56 | 1.00 | 0.00 | 0.30 |
| numeric_repro | 0.56 | 0.9333 | 0.0667 | 0.25 |
| doclocal_best | 0.55 | 0.9221 | 0.0779 | 0.23 |
| image_only | 0.57 | 0.00 | 1.00 | 0.08 |

New artifacts:

- `docs/experiment_plan_emnlp_full.md`
- `docs/ablation_tables_emnlp.md`
- `docs/ablation_tables_emnlp.csv`
- `scripts/aggregate_ablation_tables.py`
- `scripts/plot_ablation_summary.py`
- `outputs/project_profile_4090/20260417_121829/ablation_noreranker_doclocal/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/ablation_global_same_ans/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/ablation_support_strict_repro/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/ablation_support_numeric_repro/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/faithfulness_supportgate_repro/faithfulness_summary.csv`
- `outputs/project_profile_4090/20260417_121829/figures/ablation_summary_emnlp.png`
