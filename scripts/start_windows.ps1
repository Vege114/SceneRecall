[CmdletBinding()]
param(
    [string]$DataDir = $(if ($env:SCENERECALL_DATA) { $env:SCENERECALL_DATA } else { Join-Path $HOME 'SceneRecallLibrary' }),
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [string]$Python = '3.12',
    [string]$FfmpegDirectory,
    [string]$RuntimeConfig,
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
$runtimeEnvironment = @{}

function Set-RuntimeEnvironment {
    param([string]$Name, [string]$Value)
    if (-not $runtimeEnvironment.ContainsKey($Name)) {
        $runtimeEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
    }
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
}

function Find-FfmpegDirectory {
    param([string]$Directory)
    if (-not $Directory) {
        return $null
    }
    foreach ($candidate in @($Directory, (Join-Path $Directory 'bin'))) {
        if ((Test-Path -LiteralPath (Join-Path $candidate 'ffmpeg.exe') -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $candidate 'ffprobe.exe') -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

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
$previousPath = $env:PATH
$previousConsoleEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    Set-RuntimeEnvironment -Name 'PYTHONUTF8' -Value '1'
    Set-RuntimeEnvironment -Name 'PYTHONIOENCODING' -Value 'utf-8'
    # Keep the existing Linux .venv and optional model environments untouched.
    $env:UV_PROJECT_ENVIRONMENT = $venvRoot
    $env:UV_CACHE_DIR = $cacheRoot

    # A machine-local file keeps migrated cache/model paths out of tracked files.
    if (-not $RuntimeConfig) {
        $defaultRuntimeConfig = Join-Path $projectRoot '.cache\windows-runtime.json'
        if (Test-Path -LiteralPath $defaultRuntimeConfig -PathType Leaf) {
            $RuntimeConfig = $defaultRuntimeConfig
        }
    }
    if ($RuntimeConfig) {
        $runtimeSettings = Get-Content -LiteralPath $RuntimeConfig -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $runtimeSettings.environment) {
            throw 'Runtime config must contain an environment object.'
        }
        foreach ($setting in $runtimeSettings.environment.PSObject.Properties) {
            if ($setting.Name -notmatch '^(SCENERECALL_[A-Z0-9_]+|HF_HOME|TORCH_HOME|HUGGINGFACE_HUB_CACHE|HF_HUB_CACHE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)$' -or
                $setting.Value -isnot [string]) {
                throw "Unsupported runtime environment setting: $($setting.Name)"
            }
            Set-RuntimeEnvironment -Name $setting.Name -Value $setting.Value
        }
        Write-Host "Runtime config: $RuntimeConfig"
    }
    if (-not $PSBoundParameters.ContainsKey('DataDir') -and $env:SCENERECALL_DATA) {
        $DataDir = $env:SCENERECALL_DATA
    }

    if (-not $FfmpegDirectory) {
        $FfmpegDirectory = $env:SCENERECALL_FFMPEG_DIR
    }
    if ($FfmpegDirectory) {
        $ffmpegBin = Find-FfmpegDirectory -Directory $FfmpegDirectory
        if (-not $ffmpegBin) {
            throw "FFmpeg directory must contain ffmpeg.exe and ffprobe.exe (directly or in bin): $FfmpegDirectory"
        }
    }
    else {
        $ffmpegBin = Find-FfmpegDirectory -Directory (Join-Path $projectRoot '.cache\ffmpeg')
    }
    if ($ffmpegBin) {
        # Only this launcher and its children see the portable installation.
        $env:PATH = $ffmpegBin + [IO.Path]::PathSeparator + $env:PATH
    }

    if (-not $env:SCENERECALL_VISUAL_PYTHON) {
        $visualPython = Join-Path $projectRoot '.venv-visual-windows\Scripts\python.exe'
        if (Test-Path -LiteralPath $visualPython -PathType Leaf) {
            Set-RuntimeEnvironment -Name 'SCENERECALL_VISUAL_PYTHON' -Value $visualPython
        }
    }

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
    if ($env:SCENERECALL_VISUAL_PYTHON) {
        Write-Host "Visual model Python: $env:SCENERECALL_VISUAL_PYTHON"
    }

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
            $problems += "$tool is missing. Put the Windows build in .cache\ffmpeg\bin, pass -FfmpegDirectory, or add its bin directory to PATH."
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
    $env:PATH = $previousPath
    if ($env:SCENERECALL_MANAGED_CONSOLE -ne '1') {
        [Console]::OutputEncoding = $previousConsoleEncoding
    }
    foreach ($name in $runtimeEnvironment.Keys) {
        if ($null -eq $runtimeEnvironment[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $runtimeEnvironment[$name], 'Process')
        }
    }
    Pop-Location
}
