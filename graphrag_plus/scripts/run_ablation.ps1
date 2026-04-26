if (Test-Path "venv\\Scripts\\Activate.ps1") {
  . "venv\\Scripts\\Activate.ps1"
} else {
  Write-Warning "Virtual environment not found. Run: python -m venv venv; venv\\Scripts\\activate"
}

$env:TEMP = "$PWD\\.tmp"
$env:TMP = "$PWD\\.tmp"
New-Item -ItemType Directory -Force .tmp | Out-Null

python -m graphrag_plus.app.cli run_ablation
