param(
    [string]$PythonExe = "E:\anaconda\envs\vqa4090\python.exe",
    [string]$DataRoot = "data/docvqa",
    [string]$SweepConfig = "configs/sweep_docvqa_gpu_full.json",
    [int]$Repeat = 2,
    [int]$MonitorIntervalSec = 1
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = "outputs/sweeps/gpu_full_$timestamp"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$gpuCsv = "$runRoot/gpu_monitor.csv"

Write-Host "[1/4] Ensure dataset exists"
if (!(Test-Path "$DataRoot/train/qa.jsonl")) {
    Invoke-Checked "$PythonExe scripts/prepare_docvqa_dataset.py --out_root $DataRoot --max_train 1000 --max_test 200 --chunk_words 24 --negative_ratio 0.25"
}

Write-Host "[2/4] Start GPU monitor"
$monitorArgs = "scripts/gpu_monitor.py --out $gpuCsv --interval $MonitorIntervalSec"
$monitorProc = Start-Process -FilePath $PythonExe -ArgumentList $monitorArgs -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2

try {
    Write-Host "[3/4] Run high-load sweeps"
    Invoke-Checked "$PythonExe scripts/run_hparam_sweeps.py --python $PythonExe --config $SweepConfig --train_qa $DataRoot/train/qa.jsonl --train_regions $DataRoot/train/regions.jsonl --test_qa $DataRoot/test/qa.jsonl --test_regions $DataRoot/test/regions.jsonl --root_out $runRoot --repeat $Repeat"
}
finally {
    Write-Host "[4/4] Stop GPU monitor"
    if ($monitorProc -and !$monitorProc.HasExited) {
        Stop-Process -Id $monitorProc.Id -Force
    }
}

Write-Host "GPU monitor log: $gpuCsv"
Write-Host "Sweep outputs: $runRoot"
Write-Host "TensorBoard:"
Write-Host "$PythonExe -m tensorboard.main --logdir $runRoot"
