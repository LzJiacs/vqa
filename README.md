# VQA-4090 Windows 实验工程（PyTorch）

本项目提供可在 **Windows + 单卡 RTX 4090** 上直接运行的完整实验流程：
- conda 环境创建（CUDA 版 PyTorch）
- 实验数据集准备（自动生成 sample 文档 VQA 数据）
- retriever / reranker / answerability 训练
- 端到端推理与评测
- wheel + 结果文件完整打包发布

## 1. 一键创建环境（Windows PowerShell）

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./scripts/setup_win4090.ps1
```

环境名默认是 `vqa4090`。

## 2. 一键跑完整实验

```powershell
./scripts/run_full_experiment.ps1
```

默认会：
1. 自动准备 `data/sample` 数据集
2. 训练 retriever（MiniLM）
3. 训练 reranker（MiniLM）
4. 训练 answerability 分类器
5. 运行推理（默认 mock VLM）
6. 输出评测指标

结果目录：`outputs/full_run`

### 可选：接入 Qwen2.5-VL

```powershell
./scripts/run_full_experiment.ps1 -UseQwen
```

## 3. 一键打包发布

```powershell
./scripts/package_release.ps1
```

打包产物位于：
- `release/*.whl` / `release/*.tar.gz`
- `release/environment.lock.yml`
- `release/outputs/full_run/*`
- `release/vqa4090_windows_release.zip`

## 4. 手动命令（按步骤）

```powershell
conda run -n vqa4090 python scripts/prepare_data.py --sample --out data/sample
conda run -n vqa4090 python scripts/train_retriever.py --qa data/sample/qa.jsonl --regions data/sample/regions.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --output outputs/full_run/retriever
conda run -n vqa4090 python scripts/train_reranker.py --qa data/sample/qa.jsonl --regions data/sample/regions.jsonl --model cross-encoder/ms-marco-MiniLM-L-6-v2 --output outputs/full_run/reranker
conda run -n vqa4090 python scripts/train_answerability.py --qa data/sample/qa.jsonl --regions data/sample/regions.jsonl --retriever sentence-transformers/all-MiniLM-L6-v2 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --output outputs/full_run/answerability
conda run -n vqa4090 python scripts/run_infer.py --qa data/sample/qa.jsonl --regions data/sample/regions.jsonl --retriever sentence-transformers/all-MiniLM-L6-v2 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --answerability_model outputs/full_run/answerability/model.joblib --output outputs/full_run/predictions.jsonl
conda run -n vqa4090 python scripts/evaluate.py --pred outputs/full_run/predictions.jsonl --gold data/sample/qa.jsonl
```

## 5. 真实数据集 + 多组超参数实验（含 TensorBoard）

### 5.1 下载并转换 DocVQA 数据集

```powershell
E:\anaconda\envs\vqa4090\python.exe scripts/prepare_docvqa_dataset.py --out_root data/docvqa --max_train 1000 --max_test 200 --chunk_words 24 --negative_ratio 0.25
```

### 5.2 运行超参数 sweep（多次实验）

```powershell
E:\anaconda\envs\vqa4090\python.exe scripts/run_hparam_sweeps.py --python E:\anaconda\envs\vqa4090\python.exe --config configs/sweep_docvqa.json --train_qa data/docvqa/train/qa.jsonl --train_regions data/docvqa/train/regions.jsonl --test_qa data/docvqa/test/qa.jsonl --test_regions data/docvqa/test/regions.jsonl
```

也可以直接一键执行：

```powershell
./scripts/run_docvqa_sweep.ps1
```

### 5.3 查看日志与可视化

```powershell
E:\anaconda\envs\vqa4090\python.exe -m tensorboard.main --logdir outputs/sweeps
```

实验输出包含：
- 每个 run 的训练日志：`train_retriever.log / train_reranker.log / train_answerability.log / run_infer.log / evaluate.log`
- 每个 run 的配置与指标：`config.json / metrics.json`
- 实验汇总：`summary.csv / summary.json`
- TensorBoard 事件文件：`outputs/sweeps/<exp>/tensorboard/*`

## 6. 显存监控 + 高负载训练（尽量拉满 4090）

### 6.1 运行高负载 sweep（含 GPU 实时监控）

```powershell
./scripts/run_docvqa_gpu_full.ps1 -Repeat 1
```

该脚本会：
1. 启动 `scripts/gpu_monitor.py`（每秒采样一次 `nvidia-smi`）
2. 跑 `configs/sweep_docvqa_gpu_full.json` 中的高负载参数组
3. 输出 `gpu_monitor.csv` + 每个 run 全量日志 + TensorBoard

### 6.2 更长时间压满显卡

```powershell
./scripts/run_docvqa_gpu_full.ps1 -Repeat 3
```

`Repeat` 越大，持续高负载时间越长。可按需增大 `batch_size` 和 `epochs`（在 `configs/sweep_docvqa_gpu_full.json` 中调整）。

## 7. VLM 版 DocQA + ChartQA（性能提升）

### 7.1 一键跑对比（Mock vs Qwen2.5-VL）

```powershell
./scripts/run_doc_chartqa_vlm.ps1 -DocMaxTest 60 -ChartMaxTest 60
```

该流程会自动：
1. 准备 DocQA 数据：`scripts/prepare_docvqa_dataset.py`
2. 准备 ChartQA 数据：`scripts/prepare_chartqa_dataset.py`
3. 分别跑 baseline（mock）与 VLM（Qwen2.5-VL）推理：`scripts/run_vlm_eval.py`
4. 输出统一评测（DocQA 用 exact，ChartQA 额外 relaxed numeric）：`scripts/evaluate_vqa.py`

### 7.2 结果位置

- `outputs/vlm_bench/<timestamp>/docqa_mock_eval.txt`
- `outputs/vlm_bench/<timestamp>/docqa_vlm_eval.txt`
- `outputs/vlm_bench/<timestamp>/chartqa_mock_eval.txt`
- `outputs/vlm_bench/<timestamp>/chartqa_vlm_eval.txt`

## 8. 4090 项目经历版一键实验（可投论文/可面试讲）

```powershell
./scripts/run_project_profile_4090.ps1 -DocTest 80 -ChartTest 80
```

流程包括：
1. DocVQA / ChartQA 数据准备
2. BGE 检索器训练
3. Cross-Encoder 重排器训练
4. 可回答性模型训练 + 阈值自动校准
5. DocQA 与 ChartQA 的 baseline/vlm 对比评测
6. 自动生成汇总 `summary.json`

附：论文与面试叙事模板见 `docs/paper_interview_story.md`。

## 9. 模型下载策略（优先 ModelScope）

项目已默认实现：**本地路径 > ModelScope > HuggingFace 回退**。  
如需临时关闭 ModelScope（例如网络慢），可在 PowerShell 中执行：

```powershell
$env:VQA_PREFER_MODELSCOPE="0"
```
