param(
    [string]$EnvName = 'vqa4090',
    [string]$OutDir = 'outputs/full_run',
    [string]$ReleaseDir = 'release'
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

Write-Host "[1/4] Building wheel and sdist"
Invoke-Checked "conda run -n $EnvName python -m pip install --upgrade build"
Invoke-Checked "conda run -n $EnvName python -m build"

Write-Host "[2/4] Exporting conda environment spec"
conda env export -n $EnvName | Set-Content -Encoding UTF8 "$ReleaseDir/environment.lock.yml"

Write-Host "[3/4] Copying artifacts"
Copy-Item -Force "dist\*" $ReleaseDir
if (Test-Path $OutDir) {
    New-Item -ItemType Directory -Force -Path "$ReleaseDir/outputs" | Out-Null
    Copy-Item -Recurse -Force $OutDir "$ReleaseDir/outputs/full_run"
}

Write-Host "[4/4] Creating zip package"
$zipPath = Join-Path $ReleaseDir "vqa4090_windows_release.zip"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path "$ReleaseDir\*" -DestinationPath $zipPath

Write-Host "Packaged successfully: $zipPath"
