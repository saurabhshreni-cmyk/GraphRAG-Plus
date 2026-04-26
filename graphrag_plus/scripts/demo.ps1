if (Test-Path "venv\\Scripts\\Activate.ps1") {
  . "venv\\Scripts\\Activate.ps1"
} else {
  Write-Warning "Virtual environment not found. Run: python -m venv venv; venv\\Scripts\\activate"
}

$env:TEMP = "$PWD\\.tmp"
$env:TMP = "$PWD\\.tmp"
New-Item -ItemType Directory -Force .tmp | Out-Null

Write-Host "== GraphRAG++ install check ==" -ForegroundColor Cyan
python scripts\check_install.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "== GraphRAG++ ingest sample data ==" -ForegroundColor Cyan
python -m graphrag_plus.app.cli ingest --files graphrag_plus/data/sample_docs/sample1.txt graphrag_plus/data/sample_docs/sample2.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "== GraphRAG++ sample query ==" -ForegroundColor Cyan
python -m graphrag_plus.app.cli query --question "Which source contradicts the cancellation claim?" --analyst-mode
exit $LASTEXITCODE
