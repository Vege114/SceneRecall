[CmdletBinding()]
param(
    [string]$DataDir = $(if ($env:SCENERECALL_DATA) { $env:SCENERECALL_DATA } else { Join-Path $HOME 'SceneRecallLibrary' }),
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [string]$Python = '3.12',
    [switch]$Setup,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') {
    throw 'Run this launcher from Windows PowerShell or PowerShell on Windows.'
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv-windows'
$appPython = Join-Path $venvRoot 'Scripts\python.exe'
$cacheRoot = Join-Path $projectRoot '.cache\uv'
$frontendRoot = Join-Path $projectRoot 'frontend'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

Push-Location $projectRoot
$previousUvEnvironment = $env:UV_PROJECT_ENVIRONMENT
$previousUvCache = $env:UV_CACHE_DIR
try {
    # Keep the existing Linux .venv and optional model environments untouched.
    $env:UV_PROJECT_ENVIRONMENT = $venvRoot
    $env:UV_CACHE_DIR = $cacheRoot

    if ($Setup) {
        $uv = Get-Command uv.exe -ErrorAction Stop
        $npm = Get-Command npm.cmd -ErrorAction Stop
        Invoke-Checked -Executable $uv.Source -Arguments @('sync', '--frozen', '--python', $Python, '--extra', 'dev')
        Invoke-Checked -Executable $npm.Source -Arguments @('ci', '--ignore-scripts')
        Push-Location $frontendRoot
        try {
            Invoke-Checked -Executable $npm.Source -Arguments @('ci')
            Invoke-Checked -Executable $npm.Source -Arguments @('run', 'build')
        }
        finally {
            Pop-Location
        }
    }

    Write-Host 'SceneRecall runtime: native Windows'
    Write-Host "Python environment: $venvRoot"
    Write-Host "Library: $DataDir"
    Write-Host "Port: $Port"

    $problems = @()
    if (Test-Path -LiteralPath $appPython -PathType Leaf) {
        Invoke-Checked -Executable $appPython -Arguments @('--version')
        Invoke-Checked -Executable $appPython -Arguments @('-c', 'import sys,scenerecall; assert (3,12) <= sys.version_info[:2] < (3,14); print(sys.platform)')
    }
    else {
        $problems += 'Windows Python environment is missing. Run scripts\start_windows.cmd -Setup first.'
    }
    foreach ($tool in @('ffmpeg', 'ffprobe')) {
        $command = Get-Command $tool -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            Write-Host "${tool}: $($command.Source)"
        }
        else {
            $problems += "$tool is not on the Windows PATH. Install FFmpeg or add its bin directory to PATH."
        }
    }
    $codex = Get-Command codex -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($codex) {
        Write-Host "Codex CLI on PATH: $($codex.Source)"
    }
    else {
        Write-Host 'Codex CLI is not on PATH. The application also checks installed Windows Codex locations.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'dist\index.html') -PathType Leaf)) {
        $problems += 'Frontend build is missing. Run scripts\start_windows.cmd -Setup first.'
    }
    if ($problems.Count -gt 0) {
        throw ($problems -join [Environment]::NewLine)
    }
    if ($Check) {
        Write-Host 'Windows runtime checks passed. No server was started.'
        return
    }

    Write-Host 'Keep this terminal open while using SceneRecall; press Ctrl+C to stop.'
    Invoke-Checked -Executable $appPython -Arguments @('-m', 'scenerecall.main', '--data-dir', $DataDir, '--port', "$Port")
}
finally {
    $env:UV_PROJECT_ENVIRONMENT = $previousUvEnvironment
    $env:UV_CACHE_DIR = $previousUvCache
    Pop-Location
}
