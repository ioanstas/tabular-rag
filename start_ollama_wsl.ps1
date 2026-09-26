# Run from PowerShell before using the RAG pipeline in WSL.
$ErrorActionPreference = 'Stop'
$route = (& wsl.exe -d Ubuntu -- ip -4 route show default) -join ' '
if ($route -notmatch 'default via ([0-9.]+)') { throw 'Cannot determine the Windows WSL gateway.' }
$gateway = $Matches[1]
$endpoint = "http://${gateway}:11435"
try {
    $null = Invoke-RestMethod -Uri "$endpoint/api/version" -TimeoutSec 2
} catch {
    $ollamaExe = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (-not (Test-Path -LiteralPath $ollamaExe)) { throw 'Install Ollama for Windows first.' }
    $previousHost = $env:OLLAMA_HOST
    try {
        $env:OLLAMA_HOST = "${gateway}:11435"
        Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden
    } finally {
        $env:OLLAMA_HOST = $previousHost
    }
    Start-Sleep -Seconds 3
    $null = Invoke-RestMethod -Uri "$endpoint/api/version" -TimeoutSec 5
}
Write-Host 'Ollama is ready. Run this in your WSL project terminal:'
Write-Host ('export OLLAMA_BASE_URL="' + $endpoint + '"')
