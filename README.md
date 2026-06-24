# Evidence-Driven VQA

一个面向真实文档问答的可验证 VQA 项目：先检索证据，再让 VLM 基于证据回答，并在证据不足时拒答。项目重点不是单纯追求更高分，而是把“答案是否可信、是否可追溯”做成可复现实验。

本仓库沉淀了完整的工程链路：
- 文档 OCR 与版面切块
- 稠密检索与 cross-encoder 重排
- 基于 Qwen2.5-VL 的证据感知推理
- answerability / support gate 拒答机制
- DocVQA、ChartQA 的评测、可视化与消融

## 项目要解决的问题

普通 VLM 在文档图像上经常能“答对”，但很难说明答案到底来自哪里。这个项目的目标是把文档问答从“能回答”推进到“能给证据、能拒答、能审计”。

核心问题有三个：
- 文档页往往信息密集，VLM 直接全图问答不稳定。
- 仅有正确率不足以说明系统可靠，尤其是在无答案样本上容易幻觉。
- 真实业务更需要“证据链 + 拒答策略”，而不是只看单个 benchmark 分数。

## 方法概览

系统采用一个明确的 evidence-first pipeline：

1. `OCR + layout regions`
   将文档图像转成带位置的文本区域。
2. `Dense retrieval`
   用 BGE 编码器从同文档区域中召回候选证据。
3. `Cross-encoder rerank`
   对候选证据按问题相关性重排。
4. `Evidence-aware VLM inference`
   将问题、文档图像和重排后的证据一起送入 `Qwen2.5-VL-3B-Instruct`。
5. `Answerability + support gate`
   在证据不足或答案无法被证据支持时输出拒答。

这条链路的重点是：把“回答”限制在可追溯证据边界内，而不是让模型自由发挥。

## 关键结果

当前最有代表性的 doc-local retrieval + support gate 实验结果：

| 设置 | DocVQA score | Task acc with abstain | Answerable acc | Unanswerable abstain | Hallucination proxy | Evidence recall@rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global retrieval | 0.48 | 0.50 | 0.60 | 0.10 | 0.90 | 0.36 |
| doc-local retrieval | 0.54 | 0.72 | 0.675 | 0.90 | 0.10 | 0.96 |
| strict support gate | 0.55 | 0.73 | 0.6875 | 0.90 | 0.10 | 0.9067 |
| numeric support gate | 0.55 | 0.72 | 0.6875 | 0.85 | 0.15 | 0.9067 |

这组结果说明了三件事：
- 最大提升来自 `doc-local retrieval`。把证据范围限制在当前文档后，证据召回和可靠性明显改善。
- `reranker` 是必要的。只靠召回无法稳定把最有用的区域送给 VLM。
- `support gate` 让项目具备了比较清晰的“可验证回答”叙事，而不是单纯的 VLM baseline。

## 真实数据与可视化展示

### 1. DocVQA 真实页面样例

项目直接使用真实文档数据而不是手工玩具样本。下面是仓库中的 DocVQA 页面示例：

![DocVQA sample](docs/assets/docvqa_sample_page.png)

### 2. ChartQA 真实图表示例

除了文档页，项目还覆盖了 ChartQA，用来验证图表理解与数值问答能力：

![ChartQA sample](docs/assets/chartqa_sample.png)

### 3. 证据可视化

下图展示了系统在真实页面上标出的候选证据区域，这也是项目“可解释性”最直观的部分之一：

![Evidence visualization](docs/assets/evidence_visualization_docvqa.png)

### 4. Grounding baselines

这个图回答了一个关键问题：VLM 到底需要图像、文字证据，还是两者都需要。

![Grounding baselines](docs/assets/grounding_baselines_doclocal_docqa_chartqa.png)

实验结论很明确：
- `image only` 很强，但不可验证。
- `retrieved text only` 明显不够。
- `retrieved evidence + image` 在 DocQA 和 ChartQA 上都更稳，说明视觉和证据文本是互补关系。

### 5. Faithfulness / verifiability

这个图把“准确”和“可验证”分开看，是项目叙事里最关键的一张图：

![Faithfulness](docs/assets/faithfulness_supportgate.png)

它说明：即便 image-only 能得到接近的分数，也不能证明答案可由检索证据支持；support gate 则把“回答是否有证据”变成显式约束。

## 项目亮点

### 研究层面

- 把多模态文档问答重新定义为“证据驱动 + 可拒答”的问题。
- 明确区分 `accuracy` 和 `verifiability`，补上很多 VLM demo 缺少的可靠性视角。
- 给出一套较完整的 ablation：global vs doc-local、with vs without reranker、strict vs numeric support gate。

### 工程层面

- 所有核心流程都脚本化，能从数据准备一路跑到结果图表。
- 支持 DocVQA、ChartQA 和 sample 数据三条路径。
- 评测结果、阈值扫描、grounding baselines、faithfulness 分析都有固定产物输出。

## 仓库结构

```text
configs/      实验配置
data/         本地数据集缓存与转换结果
docs/         论文叙事、实验报告、展示素材
experiments/  旧版实验记录与论文图
outputs/      训练、评测、可视化产物
scripts/      数据准备、训练、推理、评测、画图脚本
src/          核心库代码
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Windows + 4090 环境可直接使用：

```powershell
./scripts/setup_win4090.ps1
```

### 2. 跑一个最小样例

```powershell
./scripts/run_full_experiment.ps1
```

默认会在 sample 数据上完成：
- retriever 训练
- reranker 训练
- answerability 训练
- 推理与评测

### 3. 跑项目主实验

```powershell
./scripts/run_project_profile_4090.ps1 -DocTest 80 -ChartTest 80
```

这个入口会完成：
- DocVQA / ChartQA 数据准备
- BGE retriever 训练
- cross-encoder reranker 训练
- answerability 校准
- VLM 推理与评测
- `summary.json` 与图表产出

## 推荐查看的结果文件

如果你只想快速理解项目，优先看这些文件：

- `docs/experiment_report_20260417.md`
- `docs/paper_interview_story.md`
- `docs/paper_tables_emnlp.md`
- `outputs/project_profile_4090/20260417_121829/summary.json`
- `outputs/project_profile_4090/20260417_121829/inference_grid_doclocal_supportgate/leaderboard.csv`
- `outputs/project_profile_4090/20260417_121829/faithfulness_supportgate/faithfulness_summary.csv`


## 当前边界

这个项目现在最强的价值在于：
- 可验证性和可拒答机制
- 文档内证据边界
- 真实数据集上的系统化实验

它还不是一个新的 foundation model，也不是单靠分数碾压通用 VLM 的项目。更准确的定位是：为多模态文档问答提供一层可追溯、可审计的 evidence layer。
