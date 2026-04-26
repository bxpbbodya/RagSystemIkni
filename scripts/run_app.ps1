$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $repoRoot "app.py"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found: $pythonExe"
}

& $pythonExe -m streamlit run $appPath @args
