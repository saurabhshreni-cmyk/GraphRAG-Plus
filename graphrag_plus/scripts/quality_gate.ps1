param(
    [switch]$SkipFrontend,
    [switch]$Fix,
    [switch]$ProbeHealth,
    [string]$PythonVersion = "3.13"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageRoot = Split-Path -Parent $scriptRoot
$repoRoot = Split-Path -Parent $packageRoot
$frontendRoot = Join-Path $repoRoot "frontend"

$env:PYTHONPATH = $repoRoot

function Resolve-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $resolved = & py "-$PythonVersion" -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return $resolved.Trim()
        }
        throw "Could not resolve Python $PythonVersion via the py launcher."
    }

    $resolved = & python -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -eq 0 -and $resolved) {
        return $resolved.Trim()
    }
    throw "Could not resolve a Python executable."
}

$pythonExe = Resolve-Python

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Test-PortListening {
    param([int]$Port)

    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

Write-Host "GraphRAG++ local quality gate" -ForegroundColor Green
Write-Host "Repo: $repoRoot"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host "Python: $pythonExe"
Write-Host "Mode: $(if ($Fix) { 'fix + verify' } else { 'verify only' })"

Invoke-Step "install check" {
    Push-Location $packageRoot
    try {
        & $pythonExe scripts/check_install.py
    }
    finally {
        Pop-Location
    }
}

if ($Fix) {
    Invoke-Step "ruff autofix" {
        Push-Location $packageRoot
        try {
            & $pythonExe -m ruff check app --fix
        }
        finally {
            Pop-Location
        }
    }
    Invoke-Step "black format" {
        Push-Location $packageRoot
        try {
            & $pythonExe -m black app
        }
        finally {
            Pop-Location
        }
    }
}

Invoke-Step "ruff check" {
    Push-Location $packageRoot
    try {
        & $pythonExe -m ruff check app
    }
    finally {
        Pop-Location
    }
}
Invoke-Step "black check" {
    Push-Location $packageRoot
    try {
        & $pythonExe -m black --check app
    }
    finally {
        Pop-Location
    }
}
Invoke-Step "mypy backend" {
    Push-Location $packageRoot
    try {
        & $pythonExe -m mypy app
    }
    finally {
        Pop-Location
    }
}
Invoke-Step "pytest backend" {
    Push-Location $packageRoot
    try {
        & $pythonExe -m pytest app/tests -q
    }
    finally {
        Pop-Location
    }
}

if (-not $SkipFrontend) {
    Invoke-Step "frontend build" {
        Push-Location $frontendRoot
        try {
            $env:CI = "1"
            cmd /c npm.cmd run build
        }
        finally {
            Pop-Location
        }
    }
}

if ($ProbeHealth -and (Test-PortListening -Port 8765)) {
    Invoke-Step "backend health probe" {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 5
        $health | ConvertTo-Json -Compress
    }
}
else {
    Write-Host ""
    Write-Host "==> backend health probe skipped: pass -ProbeHealth to check 127.0.0.1:8765" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Quality gate finished." -ForegroundColor Green
