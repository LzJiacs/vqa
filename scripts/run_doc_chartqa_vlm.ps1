param(
    [string]$PythonExe = "E:\anaconda\envs\vqa4090\python.exe",
    [string]$ModelName = "Qwen/Qwen2.5-VL-3B-Instruct",
    [int]$DocMaxTest = 200,
    [int]$ChartMaxTest = 300
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outRoot = "outputs/vlm_bench/$ts"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

Write-Host "[1/6] Prepare DocQA dataset"
Invoke-Checked "$PythonExe scripts/prepare_docvqa_dataset.py --out_root data/docvqa --max_train 1000 --max_test $DocMaxTest --chunk_words 24 --negative_ratio 0.2"

Write-Host "[2/6] Prepare ChartQA dataset"
Invoke-Checked "$PythonExe scripts/prepare_chartqa_dataset.py --out_root data/chartqa --max_train 6000 --max_val 1000 --max_test $ChartMaxTest"

Write-Host "[3/6] DocQA baseline (mock)"
Invoke-Checked "$PythonExe scripts/run_vlm_eval.py --qa data/docvqa/test/qa.jsonl --regions data/docvqa/test/regions.jsonl --backend mock --output $outRoot/docqa_mock_pred.jsonl"
Invoke-Checked "$PythonExe scripts/evaluate_vqa.py --pred $outRoot/docqa_mock_pred.jsonl --gold data/docvqa/test/qa.jsonl --mode docqa > $outRoot/docqa_mock_eval.txt"

Write-Host "[4/6] DocQA VLM (Qwen2.5-VL)"
Invoke-Checked "$PythonExe scripts/run_vlm_eval.py --qa data/docvqa/test/qa.jsonl --regions data/docvqa/test/regions.jsonl --backend qwen2.5-vl --model $ModelName --output $outRoot/docqa_vlm_pred.jsonl"
Invoke-Checked "$PythonExe scripts/evaluate_vqa.py --pred $outRoot/docqa_vlm_pred.jsonl --gold data/docvqa/test/qa.jsonl --mode docqa > $outRoot/docqa_vlm_eval.txt"

Write-Host "[5/6] ChartQA baseline (mock)"
Invoke-Checked "$PythonExe scripts/run_vlm_eval.py --qa data/chartqa/test/qa.jsonl --regions data/chartqa/test/regions.jsonl --backend mock --output $outRoot/chartqa_mock_pred.jsonl"
Invoke-Checked "$PythonExe scripts/evaluate_vqa.py --pred $outRoot/chartqa_mock_pred.jsonl --gold data/chartqa/test/qa.jsonl --mode chartqa > $outRoot/chartqa_mock_eval.txt"

Write-Host "[6/6] ChartQA VLM (Qwen2.5-VL)"
Invoke-Checked "$PythonExe scripts/run_vlm_eval.py --qa data/chartqa/test/qa.jsonl --regions data/chartqa/test/regions.jsonl --backend qwen2.5-vl --model $ModelName --output $outRoot/chartqa_vlm_pred.jsonl"
Invoke-Checked "$PythonExe scripts/evaluate_vqa.py --pred $outRoot/chartqa_vlm_pred.jsonl --gold data/chartqa/test/qa.jsonl --mode chartqa > $outRoot/chartqa_vlm_eval.txt"

Write-Host "Done. Results root: $outRoot"
Write-Host "DocQA  : $outRoot/docqa_mock_eval.txt vs $outRoot/docqa_vlm_eval.txt"
Write-Host "ChartQA: $outRoot/chartqa_mock_eval.txt vs $outRoot/chartqa_vlm_eval.txt"
