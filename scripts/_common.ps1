# Shared helpers for the Amnesia PowerShell scripts.
# This file is dot-sourced by setup.ps1, serve.ps1, chat.ps1 and doctor.ps1.
# It only defines variables/functions; it has no side effects beyond setting
# $script:AmnesiaLastExitCode (the exit code of the last CLI call).

$script:RepoRoot = Split-Path -Parent $PSScriptRoot
$script:VenvDir = Join-Path $script:RepoRoot ".venv"
$script:VenvPython = Join-Path $script:VenvDir "Scripts\python.exe"
$script:AmnesiaExe = Join-Path $script:VenvDir "Scripts\amnesia.exe"
$script:AmnesiaLastExitCode = 0

# The rich-based CLI prints Unicode glyphs (check marks, box drawing). A
# non-UTF-8 console code page (e.g. GBK/CP936) makes Python crash on redirects,
# so ask Python for UTF-8 I/O for every process these scripts start.
if (-not $env:PYTHONIOENCODING) {
    $env:PYTHONIOENCODING = "utf-8"
}

function Get-AmnesiaCli {
    <# Return the path of the installed `amnesia` executable, or $null. #>
    if (Test-Path -LiteralPath $script:AmnesiaExe) {
        return $script:AmnesiaExe
    }
    $command = Get-Command amnesia -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        return $command.Source
    }
    return $null
}

function Get-SystemPython {
    <# Return the path of the system `python` executable, or $null. #>
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        return $command.Source
    }
    return $null
}

function Get-OllamaBaseUrl {
    <# Resolve the Ollama base URL the same way Amnesia does (env var first). #>
    $raw = $env:AMNESIA_OLLAMA_URL
    if ([string]::IsNullOrWhiteSpace($raw)) {
        $raw = "http://127.0.0.1:11434"
    }
    if ($raw -notmatch "^[a-zA-Z][a-zA-Z0-9+.-]*://") {
        $raw = "http://$raw"
    }
    return $raw.TrimEnd("/")
}

function Get-OllamaStatus {
    <# Probe the Ollama daemon. Always returns a status hashtable. #>
    param([int]$TimeoutSec = 5)

    $status = @{
        Ok      = $false
        Url     = Get-OllamaBaseUrl
        Version = $null
        Models  = @()
        Error   = $null
    }
    try {
        $version = Invoke-RestMethod -Uri "$($status.Url)/api/version" -Method Get -TimeoutSec $TimeoutSec
        $status.Ok = $true
        if ($null -ne $version -and $null -ne $version.version) {
            $status.Version = [string]$version.version
        }
        $tags = Invoke-RestMethod -Uri "$($status.Url)/api/tags" -Method Get -TimeoutSec $TimeoutSec
        foreach ($model in @($tags.models)) {
            if ($null -ne $model -and $model.name) {
                $status.Models += [string]$model.name
            }
        }
    } catch {
        $status.Error = $_.Exception.Message
    }
    return $status
}

function Invoke-Amnesia {
    <#
    .SYNOPSIS
        Run the Amnesia CLI, preferring the editable install, then the global
        `amnesia` command, then running from source with an available Python.
    .DESCRIPTION
        CLI output is forwarded to the caller's console (not captured). The
        exit code is stored in $script:AmnesiaLastExitCode.
    #>
    param([string[]]$Arguments = @())

    $cli = Get-AmnesiaCli
    if ($cli) {
        & $cli @Arguments
        $script:AmnesiaLastExitCode = $LASTEXITCODE
        return
    }

    # Fallback: run the Typer app straight from the source tree. This keeps the
    # scripts usable on machines where `pip install -e .` was not possible.
    $pythons = @()
    if (Test-Path -LiteralPath $script:VenvPython) {
        $pythons += $script:VenvPython
    }
    $systemPython = Get-SystemPython
    if ($systemPython) {
        $pythons += $systemPython
    }

    $previousPythonPath = $env:PYTHONPATH
    try {
        foreach ($python in $pythons) {
            & $python -c "import fastapi, httpx, typer, rich, numpy, uvicorn" 2>$null
            if ($LASTEXITCODE -ne 0) {
                continue
            }
            if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
                $env:PYTHONPATH = $script:RepoRoot
            } else {
                $env:PYTHONPATH = "$script:RepoRoot;$previousPythonPath"
            }
            & $python -m amnesia @Arguments
            $script:AmnesiaLastExitCode = $LASTEXITCODE
            return
        }
    } finally {
        $env:PYTHONPATH = $previousPythonPath
    }

    throw "Amnesia is not installed and no usable Python was found. Run .\scripts\setup.ps1 first, or install it with: python -m pip install -e `".[dev,mcp]`""
}
