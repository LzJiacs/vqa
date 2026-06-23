$ErrorActionPreference = 'Stop'

$EnvName = 'vqa4090'
$PythonVersion = '3.10'

function Invoke-Checked {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

Write-Host "[1/5] Creating conda env: $EnvName (python=$PythonVersion)"
Invoke-Checked "conda create -n $EnvName python=$PythonVersion -y"

Write-Host "[2/5] Installing CUDA-enabled PyTorch (cu124)"
Invoke-Checked "conda run -n $EnvName python -m pip install --upgrade pip"
Invoke-Checked "conda run -n $EnvName python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124"

Write-Host "[3/5] Installing Python dependencies"
Invoke-Checked "conda run -n $EnvName python -m pip install -r requirements.windows.txt"

Write-Host "[4/5] Installing project package (editable)"
Invoke-Checked "conda run -n $EnvName python -m pip install -e ."

Write-Host "[5/5] Verifying GPU availability"
Invoke-Checked "conda run -n $EnvName python -c \"import torch; print('torch=', torch.__version__); print('cuda_runtime=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('cuda_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""

Write-Host "Done. Activate with: conda activate $EnvName"
