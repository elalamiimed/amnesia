#Requires -Version 5.1
<#
.SYNOPSIS
    Diagnose an Amnesia installation.

.DESCRIPTION
    Runs `amnesia doctor`: database path and integrity, Ollama version and
    models, embedding model availability, extracted memory counts and any
    warnings. Exits non-zero when a hard problem is detected.

.EXAMPLE
    .\scripts\doctor.ps1
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-Amnesia -Arguments @("doctor")
$exitCode = $script:AmnesiaLastExitCode

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Warning "amnesia doctor reported problems (exit code $exitCode)."
    Write-Host "  Common fixes:"
    Write-Host "    * Start Ollama (or run 'ollama serve') and check http://127.0.0.1:11434"
    Write-Host "    * Pull the embedding model: ollama pull nomic-embed-text"
    Write-Host "    * Re-run .\scripts\setup.ps1 -PullEmbedModel"
    Write-Host "    * See the Troubleshooting section of README.md"
}

exit $exitCode
