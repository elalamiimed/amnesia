#Requires -Version 5.1
<#
.SYNOPSIS
    Start the Amnesia memory proxy and dashboard on Windows.

.DESCRIPTION
    Runs `amnesia serve`: an Ollama-native + OpenAI-compatible HTTP proxy that
    injects long-term memory into every chat request, records the conversation
    and distils facts/events in the background. The Ollama daemon itself keeps
    running on port 11434 - the proxy listens on 11435.

.PARAMETER ListenHost
    Interface to bind. Default: 127.0.0.1 (use 0.0.0.0 to expose to your LAN;
    only do that on a trusted network or with AMNESIA_API_TOKEN set).

.PARAMETER Port
    Proxy port. Default: 11435.

.PARAMETER NoExtraction
    Disable automatic fact/activity extraction while the proxy runs.

.PARAMETER Reload
    Restart the proxy automatically when Python files change (development).

.PARAMETER Force
    Start even when the port is already in use.

.EXAMPLE
    .\scripts\serve.ps1

.EXAMPLE
    .\scripts\serve.ps1 -Port 11436 -NoExtraction
#>
[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 11435,
    [switch]$NoExtraction,
    [switch]$Reload,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535 (got $Port)."
}

# Everything the proxy serves is only reachable through the display host.
$displayHost = $ListenHost
if ($ListenHost -in @("0.0.0.0", "::", "[::]", "*")) {
    $displayHost = "127.0.0.1"
}

Write-Host ""
Write-Host "Amnesia proxy" -ForegroundColor Magenta

# --- Ollama daemon (informational; the proxy can start without it) ---------
$ollamaStatus = Get-OllamaStatus
if ($ollamaStatus.Ok) {
    Write-Host "  Ollama   : $($ollamaStatus.Url) (daemon stays on 11434)"
} else {
    Write-Host "  Ollama   : unreachable at $($ollamaStatus.Url)" -ForegroundColor Yellow
    Write-Host "             Starting the proxy anyway; chat requests will fail with" -ForegroundColor Yellow
    Write-Host "             HTTP 502 until the Ollama daemon is running." -ForegroundColor Yellow
}

# --- Port check ------------------------------------------------------------
$portInUse = $false
try {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
    if ($listeners) {
        $portInUse = $true
    }
} catch {
    $portInUse = $false
}
if ($portInUse -and -not $Force) {
    Write-Host ""
    Write-Warning "Port $Port is already in use - another Amnesia proxy may already be running."
    Write-Host "    Inspect it with: Get-NetTCPConnection -LocalPort $Port -State Listen"
    $answer = Read-Host "Start another listener anyway? [y/N]"
    if ($answer -notmatch "^(y|yes)$") {
        Write-Host "Aborted."
        exit 1
    }
}
if ($portInUse -and $Force) {
    Write-Warning "Port $Port is already in use; starting anyway (-Force)."
}

Write-Host ""
Write-Host "  API      : http://${displayHost}:$Port"
Write-Host "  OpenAI   : http://${displayHost}:$Port/v1"
Write-Host "  Dashboard: http://${displayHost}:$Port/amnesia/ui"
Write-Host "  Health    : http://${displayHost}:$Port/amnesia/health"
Write-Host ""
Write-Host "  Point any Ollama/OpenAI-compatible client here to give it memory."
Write-Host "  Docs: docs/INTEGRATIONS.md  -  press Ctrl+C to stop."
Write-Host ""

$cliArguments = @("serve", "--host", $ListenHost, "--port", "$Port")
if ($NoExtraction) {
    $cliArguments += "--no-extraction"
}
if ($Reload) {
    $cliArguments += "--reload"
}

Invoke-Amnesia -Arguments $cliArguments
exit $script:AmnesiaLastExitCode
