@echo off
REM Convenience wrapper for scripts\memory-mode.ps1.
REM
REM Many Windows machines use the default PowerShell execution policy
REM ("Restricted"/"AllSigned"), which refuses to run unsigned .ps1 files.
REM This .cmd launcher runs the same script with a process-scoped
REM -ExecutionPolicy Bypass, so it works without changing any system setting.
REM
REM Usage (same switches as the .ps1):
REM   .\scripts\memory-mode.cmd -Install
REM   .\scripts\memory-mode.cmd -Install -DisableOllamaAutostart
REM   .\scripts\memory-mode.cmd -Status
REM   .\scripts\memory-mode.cmd -Status -Open
REM   .\scripts\memory-mode.cmd -Start
REM   .\scripts\memory-mode.cmd -Stop
REM   .\scripts\memory-mode.cmd -Uninstall
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0memory-mode.ps1" %*
exit /b %ERRORLEVEL%
