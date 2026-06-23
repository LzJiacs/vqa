# Paper Tables for EMNLP-Style Draft


## Main DocVQA Reliability Progression


| system | score | task_acc | ans_acc | unans_abstain | halluc_proxy | ev_recall_rerank | abstain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global retrieval | 0.5 | 0.52 | 0.625 | 0.1 | 0.9 | 0.3867 | 0.05 |
| doc-local | 0.54 | 0.72 | 0.675 | 0.9 | 0.1 | 0.96 | 0.23 |
| doc-local + support gate | 0.55 | 0.73 | 0.6875 | 0.9 | 0.1 | 0.9067 | 0.3 |
| numeric support gate | 0.55 | 0.72 | 0.6875 | 0.85 | 0.15 | 0.96 | 0.23 |


## Faithfulness / Verifiability


| system | exact | correct_supported | pred_supported | unsupported_correct | answered_without_support | abstain |
| --- | --- | --- | --- | --- | --- | --- |
| supportgate_best | 0.56 | 0.56 | 1 | 0 | 0 | 0.3 |
| doclocal_v3_best | 0.55 | 0.55 | 0.9221 | 0 | 0.0779 | 0.23 |
| retrieved_evidence_image | 0.59 | 0.59 | 0.908 | 0 | 0.092 | 0.13 |
| image_only | 0.57 | 0 | 0 | 0.57 | 1 | 0.08 |


## DocQA Grounding Baselines


| baseline | score | answerable_acc | ev_recall_rerank |
| --- | --- | --- | --- |
| retrieved_evidence_image | 0.58 | 0.725 | 0.96 |
| gold_evidence_image | 0.58 | 0.725 | 1.0 |
| retrieved_image_only | 0.57 | 0.7125 | 0.96 |
| image_only | 0.57 | 0.7125 | 0.0 |
| retrieved_text_only | 0.28 | 0.35 | 0.96 |
| question_only | 0.0 | 0.0 | 0.0 |


## ChartQA Grounding Baselines


| baseline | score | relaxed | answerable_acc | ev_recall_rerank |
| --- | --- | --- | --- | --- |
| retrieved_evidence_image | 0.6 | 0.65 | 0.65 | 1.0 |
| retrieved_image_only | 0.5625 | 0.6 | 0.6 | 1.0 |
| image_only | 0.5625 | 0.6 | 0.6 | 0.0 |
| retrieved_text_only | 0.06875 | 0.1 | 0.1 | 1.0 |
| all_doc_text_only | 0.06875 | 0.1 | 0.1 | 1.0 |
| question_only | 0.03125 | 0.0625 | 0.0625 | 0.0 |
