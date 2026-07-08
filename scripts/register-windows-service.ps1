param(
    [ValidateSet("Install", "Update", "Remove", "Restart")]
    [string]$Action = "Install",

    [string]$ServiceName = "HazeWeatherRadio",
    [string]$DisplayName = "Haze Weather Radio",
    [string]$RuntimeDir = "",
    [string]$Config = "config.yaml"
)

$ErrorActionPreference = "Stop"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-RuntimeDir {
    param([string]$Value)

    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        return (Resolve-Path -LiteralPath $Value).Path
    }

    $scriptDir = Split-Path -Parent $PSCommandPath
    $candidate = Split-Path -Parent $scriptDir
    if (Test-Path -LiteralPath (Join-Path $candidate "bin\haze.exe")) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    return (Resolve-Path -LiteralPath (Get-Location)).Path
}

function Quote-Arg {
    param([string]$Value)
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Get-BinaryPath {
    param([string]$Root, [string]$ConfigPath, [string]$Name)

    $exe = Join-Path $Root "bin\haze.exe"
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "Haze executable was not found at $exe"
    }
    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        throw "Config file was not found at $ConfigPath"
    }

    return "$(Quote-Arg $exe) --service --service-name $(Quote-Arg $Name) --config $(Quote-Arg $ConfigPath) --workdir $(Quote-Arg $Root)"
}

function Invoke-Sc {
    $output = & sc.exe @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ($output -join [Environment]::NewLine)
    }
    return $output
}

if (-not (Test-Administrator)) {
    throw "Run this script from an elevated PowerShell session."
}

$root = Resolve-RuntimeDir $RuntimeDir
$configPath = if ([IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $root $Config
}
$configPath = (Resolve-Path -LiteralPath $configPath).Path
$binaryPath = Get-BinaryPath $root $configPath $ServiceName
$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue

switch ($Action) {
    "Install" {
        if ($existing) {
            throw "Service $ServiceName already exists. Use -Action Update to modify it."
        }
        New-Service `
            -Name $ServiceName `
            -DisplayName $DisplayName `
            -BinaryPathName $binaryPath `
            -StartupType Automatic `
            -Description "Runs the Haze Weather Radio runtime." | Out-Null
        Invoke-Sc failure $ServiceName reset= 86400 actions= restart/5000/restart/15000/restart/60000 | Out-Null
        Invoke-Sc failureflag $ServiceName 1 | Out-Null
        Start-Service -Name $ServiceName
        Write-Host "Installed and started $ServiceName as LocalSystem."
    }
    "Update" {
        if (-not $existing) {
            throw "Service $ServiceName does not exist. Use -Action Install first."
        }
        if ($existing.Status -ne "Stopped") {
            Stop-Service -Name $ServiceName -Force
            $existing.WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
        }
        Invoke-Sc config $ServiceName binPath= $binaryPath start= auto DisplayName= $DisplayName | Out-Null
        Invoke-Sc description $ServiceName "Runs the Haze Weather Radio runtime." | Out-Null
        Invoke-Sc failure $ServiceName reset= 86400 actions= restart/5000/restart/15000/restart/60000 | Out-Null
        Invoke-Sc failureflag $ServiceName 1 | Out-Null
        Start-Service -Name $ServiceName
        Write-Host "Updated and restarted $ServiceName."
    }
    "Restart" {
        if (-not $existing) {
            throw "Service $ServiceName does not exist."
        }
        Restart-Service -Name $ServiceName -Force
        Write-Host "Restarted $ServiceName."
    }
    "Remove" {
        if (-not $existing) {
            Write-Host "Service $ServiceName is not installed."
            return
        }
        if ($existing.Status -ne "Stopped") {
            Stop-Service -Name $ServiceName -Force
            $existing.WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
        }
        Invoke-Sc delete $ServiceName | Out-Null
        Write-Host "Removed $ServiceName."
    }
}
