param(
    [string]$EnvName = 'vqa4090',
    [string]$DataDir = 'data/sample',
    [string]$OutDir = 'outputs/full_run',
    [string]$RetrieverModel = 'sentence-transformers/all-MiniLM-L6-v2',
    [string]$RerankerModel = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    [switch]$UseQwen
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

Write-Host "[1/6] Preparing sample dataset"
Invoke-Checked "conda run -n $EnvName python scripts/prepare_data.py --sample --out $DataDir"

Write-Host "[2/6] Training retriever"
Invoke-Checked "conda run -n $EnvName python scripts/train_retriever.py --qa `"$DataDir/qa.jsonl`" --regions `"$DataDir/regions.jsonl`" --model $RetrieverModel --epochs 1 --batch_size 8 --output `"$OutDir/retriever`""

Write-Host "[3/6] Training reranker"
Invoke-Checked "conda run -n $EnvName python scripts/train_reranker.py --qa `"$DataDir/qa.jsonl`" --regions `"$DataDir/regions.jsonl`" --model $RerankerModel --epochs 1 --batch_size 8 --output `"$OutDir/reranker`""

Write-Host "[4/6] Training answerability classifier"
Invoke-Checked "conda run -n $EnvName python scripts/train_answerability.py --qa `"$DataDir/qa.jsonl`" --regions `"$DataDir/regions.jsonl`" --retriever $RetrieverModel --reranker $RerankerModel --retrieve_top_k 5 --output `"$OutDir/answerability`""

Write-Host "[5/6] Running inference"
if ($UseQwen) {
    Invoke-Checked "conda run -n $EnvName python scripts/run_infer.py --qa `"$DataDir/qa.jsonl`" --regions `"$DataDir/regions.jsonl`" --retriever $RetrieverModel --reranker $RerankerModel --answerability_model `"$OutDir/answerability/model.joblib`" --vlm_backend qwen2.5-vl --vlm_model Qwen/Qwen2.5-VL-3B-Instruct --output `"$OutDir/predictions.jsonl`""
} else {
    Invoke-Checked "conda run -n $EnvName python scripts/run_infer.py --qa `"$DataDir/qa.jsonl`" --regions `"$DataDir/regions.jsonl`" --retriever $RetrieverModel --reranker $RerankerModel --answerability_model `"$OutDir/answerability/model.joblib`" --vlm_backend mock --output `"$OutDir/predictions.jsonl`""
}

Write-Host "[6/6] Evaluating"
Invoke-Checked "conda run -n $EnvName python scripts/evaluate.py --pred `"$OutDir/predictions.jsonl`" --gold `"$DataDir/qa.jsonl`""

Write-Host "Full experiment complete. Outputs at: $OutDir"
