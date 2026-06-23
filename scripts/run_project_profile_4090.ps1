param(
    [string]$PythonExe = "E:\anaconda\envs\vqa4090\python.exe",
    [string]$ProfileConfig = "configs/profile_4090.json",
    [int]$DocTrain = 1000,
    [int]$DocVal = 120,
    [int]$DocTest = 80,
    [int]$DocChunkWords = 24,
    [int]$ChartTrain = 6000,
    [int]$ChartVal = 1000,
    [int]$ChartTest = 80
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Checked {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

$profile = Get-Content $ProfileConfig | ConvertFrom-Json
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outRoot = "outputs/project_profile_4090/$ts"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$retriever = $profile.docqa.retriever_model
$reranker = $profile.docqa.reranker_model
$vlm = $profile.docqa.vlm_model
$reEpoch = $profile.docqa.retriever_epochs
$rrEpoch = $profile.docqa.reranker_epochs
$bs = $profile.docqa.batch_size
$rtop = $profile.docqa.retrieve_top_k
$ktop = $profile.docqa.rerank_top_k

Write-Host "[1/12] Prepare DocVQA + ChartQA datasets"
Invoke-Checked "$PythonExe scripts/prepare_docvqa_dataset.py --out_root data/docvqa --max_train $DocTrain --max_val $DocVal --max_test $DocTest --chunk_words $DocChunkWords --negative_ratio 0.25"
Invoke-Checked "$PythonExe scripts/prepare_chartqa_dataset.py --out_root data/chartqa --max_train $ChartTrain --max_val $ChartVal --max_test $ChartTest"

Write-Host "[2/12] Train retriever (BGE)"
Invoke-Checked "$PythonExe scripts/train_retriever.py --qa data/docvqa/train/qa.jsonl --regions data/docvqa/train/regions.jsonl --model $retriever --epochs $reEpoch --batch_size $bs --output $outRoot/retriever"

Write-Host "[3/12] Train reranker (cross-encoder)"
Invoke-Checked "$PythonExe scripts/train_reranker.py --qa data/docvqa/train/qa.jsonl --regions data/docvqa/train/regions.jsonl --model $reranker --epochs $rrEpoch --batch_size $bs --output $outRoot/reranker"

Write-Host "[4/12] Train answerability classifier"
Invoke-Checked "$PythonExe scripts/train_answerability.py --qa data/docvqa/train/qa.jsonl --regions data/docvqa/train/regions.jsonl --retriever $outRoot/retriever --reranker $outRoot/reranker --retrieve_top_k $rtop --output $outRoot/answerability"

Write-Host "[5/12] Calibrate abstain threshold on DocVQA dev"
Invoke-Checked "$PythonExe scripts/calibrate_answerability_threshold.py --qa data/docvqa/val/qa.jsonl --regions data/docvqa/val/regions.jsonl --retriever $outRoot/retriever --reranker $outRoot/reranker --answerability_model $outRoot/answerability/model.joblib --retrieve_top_k $rtop --output $outRoot/abstain_threshold.json"

Write-Host "[6/12] DocQA baseline (mock)"
Invoke-Checked "$PythonExe scripts/run_infer.py --qa data/docvqa/test/qa.jsonl --regions data/docvqa/test/regions.jsonl --retriever $outRoot/retriever --reranker $outRoot/reranker --answerability_model $outRoot/answerability/model.joblib --abstain_threshold_file $outRoot/abstain_threshold.json --retrieve_top_k $rtop --rerank_top_k $ktop --vlm_backend mock --output $outRoot/docqa_mock_pred.jsonl"
$docMockEval = & $PythonExe scripts/evaluate_vqa.py --pred "$outRoot/docqa_mock_pred.jsonl" --gold data/docvqa/test/qa.jsonl --mode docqa
$docMockEval | Set-Content -Encoding UTF8 "$outRoot/docqa_mock_eval.txt"

Write-Host "[7/12] DocQA VLM (Qwen2.5-VL)"
Invoke-Checked "$PythonExe scripts/run_infer.py --qa data/docvqa/test/qa.jsonl --regions data/docvqa/test/regions.jsonl --retriever $outRoot/retriever --reranker $outRoot/reranker --answerability_model $outRoot/answerability/model.joblib --abstain_threshold_file $outRoot/abstain_threshold.json --retrieve_top_k $rtop --rerank_top_k $ktop --vlm_backend qwen2.5-vl --vlm_model $vlm --output $outRoot/docqa_vlm_pred.jsonl"
$docVlmEval = & $PythonExe scripts/evaluate_vqa.py --pred "$outRoot/docqa_vlm_pred.jsonl" --gold data/docvqa/test/qa.jsonl --mode docqa
$docVlmEval | Set-Content -Encoding UTF8 "$outRoot/docqa_vlm_eval.txt"
Invoke-Checked "$PythonExe scripts/evaluate_reliability.py --pred $outRoot/docqa_vlm_pred.jsonl --gold data/docvqa/test/qa.jsonl --mode docqa --output $outRoot/docqa_vlm_reliability.json"
Invoke-Checked "$PythonExe scripts/visualize_evidence.py --pred $outRoot/docqa_vlm_pred.jsonl --gold data/docvqa/test/qa.jsonl --regions data/docvqa/test/regions.jsonl --output_dir $outRoot/evidence_vis/docqa --max_images 16"

Write-Host "[8/12] ChartQA baseline (mock)"
Invoke-Checked "$PythonExe scripts/run_vlm_eval.py --qa data/chartqa/test/qa.jsonl --regions data/chartqa/test/regions.jsonl --backend mock --output $outRoot/chartqa_mock_pred.jsonl"
$chartMockEval = & $PythonExe scripts/evaluate_vqa.py --pred "$outRoot/chartqa_mock_pred.jsonl" --gold data/chartqa/test/qa.jsonl --mode chartqa
$chartMockEval | Set-Content -Encoding UTF8 "$outRoot/chartqa_mock_eval.txt"

Write-Host "[9/12] ChartQA VLM (Qwen2.5-VL)"
Invoke-Checked "$PythonExe scripts/run_vlm_eval.py --qa data/chartqa/test/qa.jsonl --regions data/chartqa/test/regions.jsonl --backend qwen2.5-vl --model $vlm --output $outRoot/chartqa_vlm_pred.jsonl"
$chartVlmEval = & $PythonExe scripts/evaluate_vqa.py --pred "$outRoot/chartqa_vlm_pred.jsonl" --gold data/chartqa/test/qa.jsonl --mode chartqa
$chartVlmEval | Set-Content -Encoding UTF8 "$outRoot/chartqa_vlm_eval.txt"
Invoke-Checked "$PythonExe scripts/evaluate_reliability.py --pred $outRoot/chartqa_vlm_pred.jsonl --gold data/chartqa/test/qa.jsonl --mode chartqa --output $outRoot/chartqa_vlm_reliability.json"

Write-Host "[10/12] Generate summary"
Invoke-Checked "$PythonExe scripts/summarize_project_results.py --root $outRoot --output $outRoot/summary.json"

Write-Host "[11/12] Record profile"
Copy-Item -Force -Path $ProfileConfig -Destination "$outRoot/profile.json"
New-Item -ItemType Directory -Force -Path "$outRoot/data_snapshot/docvqa" | Out-Null
New-Item -ItemType Directory -Force -Path "$outRoot/data_snapshot/chartqa" | Out-Null
Copy-Item -Force -Path "data/docvqa/train/qa.jsonl" -Destination "$outRoot/data_snapshot/docvqa/train_qa.jsonl"
Copy-Item -Force -Path "data/docvqa/train/regions.jsonl" -Destination "$outRoot/data_snapshot/docvqa/train_regions.jsonl"
Copy-Item -Force -Path "data/docvqa/val/qa.jsonl" -Destination "$outRoot/data_snapshot/docvqa/val_qa.jsonl"
Copy-Item -Force -Path "data/docvqa/val/regions.jsonl" -Destination "$outRoot/data_snapshot/docvqa/val_regions.jsonl"
Copy-Item -Force -Path "data/docvqa/test/qa.jsonl" -Destination "$outRoot/data_snapshot/docvqa/test_qa.jsonl"
Copy-Item -Force -Path "data/docvqa/test/regions.jsonl" -Destination "$outRoot/data_snapshot/docvqa/test_regions.jsonl"
Copy-Item -Force -Path "data/chartqa/test/qa.jsonl" -Destination "$outRoot/data_snapshot/chartqa/test_qa.jsonl"
Copy-Item -Force -Path "data/chartqa/test/regions.jsonl" -Destination "$outRoot/data_snapshot/chartqa/test_regions.jsonl"

Write-Host "[12/12] Done"
Write-Host "Done. Results root: $outRoot"
