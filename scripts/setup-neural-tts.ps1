param(
    [string] $Python = "py -3.12",
    [switch] $SkipF5,
    [switch] $SkipChatterbox
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
$Venv = Join-Path $Root "managed/venvs/neural-tts"
$PythonExe = Join-Path $Venv "Scripts/python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    $parts = $Python -split '\s+'
    $exe = $parts[0]
    $args = @()
    if ($parts.Count -gt 1) {
        $args = $parts[1..($parts.Count - 1)]
    }
    & $exe @args -m venv $Venv
}

& $PythonExe -m pip install --upgrade pip wheel setuptools
& $PythonExe -m pip install torch torchaudio
if (-not $SkipF5) {
    & $PythonExe -m pip install f5-tts
}
if (-not $SkipChatterbox) {
    & $PythonExe -m pip install chatterbox-tts
}

Write-Host "Neural TTS runtime ready: $PythonExe"
Write-Host "Haze will use this venv automatically when provider='f5tts' or provider='chatterbox'."
