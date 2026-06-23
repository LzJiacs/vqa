param(
    [string]$PythonExe = "E:\anaconda\envs\vqa4090\python.exe",
    [string]$DataRoot = "data/docvqa",
    [int]$MaxTrain = 1000,
    [int]$MaxTest = 200,
    [int]$ChunkWords = 24,
    [double]$NegativeRatio = 0.25
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

Write-Host "[1/2] Downloading + converting DocVQA dataset"
Invoke-Checked "$PythonExe scripts/prepare_docvqa_dataset.py --out_root $DataRoot --max_train $MaxTrain --max_test $MaxTest --chunk_words $ChunkWords --negative_ratio $NegativeRatio"

Write-Host "[2/2] Running hyper-parameter sweeps with TensorBoard logging"
Invoke-Checked "$PythonExe scripts/run_hparam_sweeps.py --python $PythonExe --config configs/sweep_docvqa.json --train_qa $DataRoot/train/qa.jsonl --train_regions $DataRoot/train/regions.jsonl --test_qa $DataRoot/test/qa.jsonl --test_regions $DataRoot/test/regions.jsonl"

Write-Host "Done. Open TensorBoard with:"
Write-Host "$PythonExe -m tensorboard.main --logdir outputs/sweeps"
