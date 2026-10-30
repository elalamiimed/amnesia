#Requires -Version 5.1
<#
.SYNOPSIS
    Start the Amnesia memory proxy and dashboard on Windows.

.DESCRIPTION
    Runs `amnesia serve`: an Ollama-native + OpenAI-compatible HTTP proxy that
    injects long-term memory into every chat request, records the conversation
    and distils facts/events in the background.

    By default the Ollama daemon keeps running on port 11434 and the proxy
    listens on 11435; you then point clients at http://127.0.0.1:11435 (or set
    OLLAMA_HOST=127.0.0.1:11435 in a shell for `ollama run`).

    With -Takeover Amnesia binds Ollama's own default port 11434 instead and
    moves the real Ollama daemon to a private loopback port (11435 by default)
    that it manages as a child process. Plain `ollama run` - and any client that
    defaults to 11434 - then gets memory injection and conversation capture with
    NO OLLAMA_HOST configuration. Takeover stops the Ollama desktop app or
    `ollama serve` daemon that owns 11434 first (unless -KeepOllama), runs the
    private daemon itself, and restores the desktop app on Ctrl+C (unless
    -NoRestartOllama). It needs the optional `psutil` package:
        python -m pip install psutil

.PARAMETER ListenHost
    Interface to bind. Default: 127.0.0.1 (use 0.0.0.0 to expose to your LAN;
    only do that on a trusted network or with AMNESIA_API_TOKEN set).

.PARAMETER Port
    Proxy port. Default: 11435. In takeover mode Amnesia binds Ollama's standard
    port 11434, so only pass -Port explicitly if you really mean it (plain
    clients would not be captured on a non-default port).

.PARAMETER NoExtraction
    Disable automatic fact/activity extraction while the proxy runs.

.PARAMETER Reload
    Restart the proxy automatically when Python files change (development).
    Cannot be combined with -Takeover.

.PARAMETER Force
    Start even when the proxy port is already in use (proxy mode only).

.PARAMETER Takeover
    Zero-config mode: take over Ollama's default port 11434, manage the real
    Ollama daemon on a private port, and give plain `ollama run` memory.

.PARAMETER UpstreamPort
    Private port for the managed Ollama daemon in takeover mode. Default: 11435.

.PARAMETER NoRestartOllama
    In takeover mode, do not restart the Ollama desktop app afterwards.

.PARAMETER KeepOllama
    In takeover mode, fail instead of stopping an Ollama app/daemon already
    running on 11434.

.EXAMPLE
    .\scripts\serve.ps1

.EXAMPLE
    .\scripts\serve.ps1 -Port 11436 -NoExtraction

.EXAMPLE
    .\scripts\serve.ps1 -Takeover

.EXAMPLE
    .\scripts\serve.ps1 -Takeover -UpstreamPort 11436 -NoRestartOllama
#>
[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 11435,
    [switch]$NoExtraction,
    [switch]$Reload,
    [switch]$Force,
    [switch]$Takeover,
    [int]$UpstreamPort = 11435,
    [switch]$NoRestartOllama,
    [switch]$KeepOllama
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

function Test-PortOpen {
    <#
    .SYNOPSIS
        Return $true when something accepts a TCP connection on host:port.
    .DESCRIPTION
        A plain socket probe - more reliable than Get-NetTCPConnection on
        locked-down or virtualised Windows hosts where the CIM cmdlet can fail
        silently. Used for port preflight checks only.
    #>
    param(
        [string]$TargetHost,
        [int]$TargetPort,
        [int]$TimeoutMs = 500
    )
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($TargetHost, $TargetPort)
        if (-not $task.Wait($TimeoutMs)) {
            return $false
        }
        return $client.Connected
    } catch {
        return $false
    } finally {
        try { $client.Dispose() } catch { }
    }
}

# Takeover binds Ollama's own default port unless -Port was explicitly given.
$defaultOllamaPort = 11434
$portExplicitlySet = $PSBoundParameters.ContainsKey("Port")
$takeoverPort = if ($portExplicitlySet) { $Port } else { $defaultOllamaPort }

if ($Takeover) {
    if ($Reload) {
        throw "-Takeover cannot be combined with -Reload."
    }
    if ($takeoverPort -lt 1 -or $takeoverPort -gt 65535) {
        throw "Takeover port must be between 1 and 65535 (got $takeoverPort)."
    }
    if ($UpstreamPort -lt 1 -or $UpstreamPort -gt 65535) {
        throw "Upstream port must be between 1 and 65535 (got $UpstreamPort)."
    }
    if ($UpstreamPort -eq $takeoverPort) {
        throw "-UpstreamPort ($UpstreamPort) must differ from the port Amnesia binds ($takeoverPort)."
    }
} else {
    if ($Port -lt 1 -or $Port -gt 65535) {
        throw "Port must be between 1 and 65535 (got $Port)."
    }
}

# Everything the proxy serves is only reachable through the display host.
$displayHost = $ListenHost
if ($ListenHost -in @("0.0.0.0", "::", "[::]", "*")) {
    $displayHost = "127.0.0.1"
}

Write-Host ""
if ($Takeover) {
    Write-Host "Amnesia takeover mode" -ForegroundColor Magenta
    Write-Host "  Ollama clients   : http://${displayHost}:${takeoverPort}  (plain ``ollama run`` - no OLLAMA_HOST)"
    Write-Host "  Private daemon   : http://127.0.0.1:${UpstreamPort}  (managed by Amnesia)"
    Write-Host "  Dashboard        : http://${displayHost}:${takeoverPort}/amnesia/ui"
    Write-Host "  Daemon log       : $(Join-Path $HOME '.amnesia\logs\ollama-takeover.log')"
    Write-Host ""
    Write-Host "  Takeover stops the Ollama app/daemon that owns port ${takeoverPort}," -ForegroundColor Yellow
    Write-Host "  runs Ollama privately on ${UpstreamPort}, and restores the Ollama" -ForegroundColor Yellow
    Write-Host "  desktop app on Ctrl+C. Do not launch the desktop app while takeover runs." -ForegroundColor Yellow
    if ($displayHost -eq "127.0.0.1") {
        Write-Host "  Bind address     : 127.0.0.1 (loopback only)" -ForegroundColor DarkGray
    } else {
        Write-Host "  Bind address     : $ListenHost (exposed to the network - make sure this is intended)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Amnesia proxy" -ForegroundColor Magenta
}

# --- Optional dependency for takeover mode (psutil) ------------------------
if ($Takeover) {
    $pythonForCheck = $null
    if (Test-Path -LiteralPath $script:VenvPython) {
        $pythonForCheck = $script:VenvPython
    } else {
        $pythonForCheck = Get-SystemPython
    }
    if ($pythonForCheck) {
        & $pythonForCheck -c "import psutil" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Takeover mode needs psutil. Install it with: python -m pip install psutil"
        }
    }
}

# --- Ollama daemon + port checks ------------------------------------------
$ollamaStatus = Get-OllamaStatus

if ($Takeover) {
    if ($ollamaStatus.Ok) {
        Write-Host "  Ollama           : detected at $($ollamaStatus.Url) (version $($ollamaStatus.Version))"
        Write-Host "                     -> it will be stopped and run privately on port $UpstreamPort."
    } else {
        Write-Host "  Ollama           : not answering at $($ollamaStatus.Url)" -ForegroundColor Yellow
        Write-Host "                     -> takeover will start its own managed daemon on port $UpstreamPort."
    }

    $upstreamInUse = Test-PortOpen -TargetHost "127.0.0.1" -TargetPort $UpstreamPort
    if ($upstreamInUse) {
        throw "Upstream port $UpstreamPort is already in use. Pick a free one with -UpstreamPort <port> (for example 11436)."
    }

    if ($KeepOllama -and $ollamaStatus.Ok) {
        Write-Host "  KeepOllama       : set - takeover will stop if it cannot leave Ollama alone." -ForegroundColor Yellow
    }
    if ($Force) {
        Write-Host "  Note             : -Force only affects proxy mode and is ignored in takeover mode." -ForegroundColor DarkGray
    }
} else {
    # --- Ollama daemon (informational; the proxy can start without it) -----
    if ($ollamaStatus.Ok) {
        Write-Host "  Ollama   : $($ollamaStatus.Url) (daemon stays on 11434)"
    } else {
        Write-Host "  Ollama   : unreachable at $($ollamaStatus.Url)" -ForegroundColor Yellow
        Write-Host "             Starting the proxy anyway; chat requests will fail with" -ForegroundColor Yellow
        Write-Host "             HTTP 502 until the Ollama daemon is running." -ForegroundColor Yellow
    }

    # --- Port check -------------------------------------------------------
    $portInUse = Test-PortOpen -TargetHost $displayHost -TargetPort $Port
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
}

$cliArguments = @("serve")
if ($Takeover) {
    $cliArguments += @("--takeover", "--host", $ListenHost, "--upstream-port", "$UpstreamPort")
    if ($portExplicitlySet) {
        $cliArguments += @("--port", "$Port")
    }
    if ($NoRestartOllama) {
        $cliArguments += "--no-restart-ollama"
    }
    if ($KeepOllama) {
        $cliArguments += "--keep-ollama"
    }
} else {
    $cliArguments += @("--host", $ListenHost, "--port", "$Port")
}
if ($NoExtraction) {
    $cliArguments += "--no-extraction"
}
if ($Reload) {
    $cliArguments += "--reload"
}

Invoke-Amnesia -Arguments $cliArguments
exit $script:AmnesiaLastExitCode
