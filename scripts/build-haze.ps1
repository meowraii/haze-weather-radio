param(
    [ValidateSet("debug", "release")]
    [string] $Profile = "release",
    [string] $OutputDir = "",
    [ValidateSet("builtin", "rsmpeg")]
    [string] $MediaBackend = "rsmpeg",
    [switch] $IncludeEnv,
    [switch] $SkipGoServices,
    [switch] $SkipCargoBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")

function Get-PortableOSName {
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
        return "Windows"
    }
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)) {
        return "Linux"
    }
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) {
        return "macOS"
    }
    return "UnknownOS"
}

function Get-PortableArchName {
    switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
        "X64" { return "x86_64" }
        "Arm64" { return "aarch64" }
        "X86" { return "x86" }
        "Arm" { return "armv7" }
        default { return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant() }
    }
}

function Get-DefaultPortableOutputDir {
    return "dist/Haze_UAP-$(Get-PortableOSName)-$(Get-PortableArchName)-Portable"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Get-DefaultPortableOutputDir
}

$DistRoot = [System.IO.Path]::GetFullPath((Join-Path $Root "dist"))
$OutFull = [System.IO.Path]::GetFullPath((Join-Path $Root $OutputDir))
$BinFull = Join-Path $OutFull "bin"
$DistRootWithSeparator = $DistRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar

if ($OutFull -ne $DistRoot -and -not $OutFull.StartsWith($DistRootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside the dist directory: $OutFull"
}

function Copy-BundleDirectory {
    param(
        [Parameter(Mandatory = $true)][string] $Name
    )
    $Source = Join-Path $Root (Join-Path "bundle" $Name)
    $Target = Join-Path $OutFull $Name
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        return
    }
    $PreserveDir = $null
    if ($Name -eq "managed" -and (Test-Path -LiteralPath $Target -PathType Container)) {
        $OnnxFiles = @(Get-ChildItem -LiteralPath $Target -Filter "*.onnx" -Recurse -File -ErrorAction SilentlyContinue)
        if ($OnnxFiles.Count -gt 0) {
            $PreserveDir = Join-Path ([System.IO.Path]::GetTempPath()) ("haze-preserve-onnx-" + [System.Guid]::NewGuid().ToString("N"))
            $TargetFull = [System.IO.Path]::GetFullPath($Target).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
            foreach ($File in $OnnxFiles) {
                $Relative = $File.FullName.Substring($TargetFull.Length).TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
                $Backup = Join-Path $PreserveDir $Relative
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Backup) | Out-Null
                Copy-Item -LiteralPath $File.FullName -Destination $Backup -Force
            }
        }
    }
    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
    Copy-Item -LiteralPath $Source -Destination $Target -Recurse -Force
    if ($PreserveDir -and (Test-Path -LiteralPath $PreserveDir -PathType Container)) {
        $PreserveFull = [System.IO.Path]::GetFullPath($PreserveDir).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
        Get-ChildItem -LiteralPath $PreserveDir -Filter "*.onnx" -Recurse -File | ForEach-Object {
            $Relative = $_.FullName.Substring($PreserveFull.Length).TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
            $Restored = Join-Path $Target $Relative
            if (-not (Test-Path -LiteralPath $Restored)) {
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Restored) | Out-Null
                Copy-Item -LiteralPath $_.FullName -Destination $Restored -Force
            }
        }
        Remove-Item -LiteralPath $PreserveDir -Recurse -Force
    }
}

function Test-RunningOnWindows {
    return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
}

function Get-MsysRoot {
    if ([string]::IsNullOrWhiteSpace($env:MSYS2_ROOT)) {
        return "C:\msys64"
    }
    return $env:MSYS2_ROOT
}

function Initialize-Clang64BuildEnvironment {
    $MsysRoot = Get-MsysRoot
    $MsysUsrBin = Join-Path $MsysRoot "usr\bin"
    $Clang64Root = Join-Path $MsysRoot "clang64"
    $Clang64Bin = Join-Path $Clang64Root "bin"
    $Clang64Lib = Join-Path $Clang64Root "lib"
    foreach ($RequiredTool in @("x86_64-w64-mingw32-clang.exe", "x86_64-w64-mingw32-clang++.exe", "llvm-ar.exe", "pkg-config.exe", "llvm-objdump.exe")) {
        $RequiredPath = Join-Path $Clang64Bin $RequiredTool
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "required MSYS2 CLANG64 tool not found: $RequiredPath"
        }
    }

    $Clang = Join-Path $Clang64Bin "x86_64-w64-mingw32-clang.exe"
    $Clangxx = Join-Path $Clang64Bin "x86_64-w64-mingw32-clang++.exe"
    $env:Path = "$Clang64Bin;$MsysUsrBin;$env:Path"
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER = $Clang
    $env:CC_x86_64_pc_windows_gnullvm = $Clang
    $env:CXX_x86_64_pc_windows_gnullvm = $Clangxx
    $env:AR_x86_64_pc_windows_gnullvm = Join-Path $Clang64Bin "llvm-ar.exe"
    $env:LIBCLANG_PATH = $Clang64Bin
    $env:PKG_CONFIG = Join-Path $Clang64Bin "pkg-config.exe"
    $env:PKG_CONFIG_PATH = "$(Join-Path $Clang64Lib "pkgconfig");$(Join-Path $Clang64Root "share\pkgconfig")"
}

function New-FFmpegIncludeOverlay {
    param(
        [Parameter(Mandatory = $true)][string] $SourceInclude,
        [Parameter(Mandatory = $true)][string] $Name
    )

    if (-not (Test-Path -LiteralPath $SourceInclude -PathType Container)) {
        throw "FFmpeg include directory not found: $SourceInclude"
    }

    $OverlayDir = Join-Path $Root "target/ffmpeg-include-overlay/$Name"
    if (Test-Path -LiteralPath $OverlayDir) {
        Remove-Item -LiteralPath $OverlayDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $OverlayDir | Out-Null
    Copy-Item -Path (Join-Path $SourceInclude "*") -Destination $OverlayDir -Recurse -Force

    $AvcodecOverlay = Join-Path $OverlayDir "libavcodec"
    New-Item -ItemType Directory -Force -Path $AvcodecOverlay | Out-Null
    $Avfft = Join-Path $AvcodecOverlay "avfft.h"
    if (-not (Test-Path -LiteralPath $Avfft -PathType Leaf)) {
        @"
#ifndef HAZE_COMPAT_LIBAVCODEC_AVFFT_H
#define HAZE_COMPAT_LIBAVCODEC_AVFFT_H
#include <libavcodec/avcodec.h>
#endif
"@ | Set-Content -LiteralPath $Avfft -Encoding ASCII
    }

    return $OverlayDir
}

function Get-RustyFfmpegBindingPath {
    $CargoHome = if ([string]::IsNullOrWhiteSpace($env:CARGO_HOME)) {
        Join-Path $env:USERPROFILE ".cargo"
    } else {
        $env:CARGO_HOME
    }
    $RegistrySrc = Join-Path $CargoHome "registry\src"
    if (-not (Test-Path -LiteralPath $RegistrySrc -PathType Container)) {
        return ""
    }

    $Candidate = Get-ChildItem -LiteralPath $RegistrySrc -Recurse -Filter "binding.rs" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "rusty_ffmpeg-0\.16\.7\+ffmpeg\.8[\\/]+src[\\/]+binding\.rs$" } |
        Select-Object -First 1
    if ($null -eq $Candidate) {
        return ""
    }
    return $Candidate.FullName
}

function Initialize-Clang64RsmpegBuildEnvironment {
    Initialize-Clang64BuildEnvironment

    $MsysRoot = Get-MsysRoot
    $Clang64Root = Join-Path $MsysRoot "clang64"
    $Clang64Bin = Join-Path $Clang64Root "bin"
    $Clang64Include = Join-Path $Clang64Root "include"
    $Clang64Lib = Join-Path $Clang64Root "lib"
    foreach ($RequiredPath in @(
        (Join-Path $Clang64Include "libavcodec\avcodec.h"),
        (Join-Path $Clang64Lib "libavcodec.dll.a"),
        (Join-Path $Clang64Bin "avcodec-62.dll"),
        (Join-Path $Clang64Bin "avformat-62.dll"),
        (Join-Path $Clang64Bin "avutil-60.dll")
    )) {
        if (-not (Test-Path -LiteralPath $RequiredPath)) {
            throw "required MSYS2 CLANG64 FFmpeg file not found: $RequiredPath"
        }
    }

    $OverlayDir = New-FFmpegIncludeOverlay -SourceInclude $Clang64Include -Name "clang64"
    $BindingPath = Get-RustyFfmpegBindingPath
    if ([string]::IsNullOrWhiteSpace($BindingPath)) {
        throw "rusty_ffmpeg FFmpeg 8 binding.rs was not found in the Cargo registry. Run `cargo fetch` and retry."
    }

    $env:FFMPEG_INCLUDE_DIR = $OverlayDir
    $env:FFMPEG_LIBS_DIR = $Clang64Lib
    $env:FFMPEG_DLL_PATH = $Clang64Bin
    $env:FFMPEG_LINK_MODE = "dynamic"
    $env:FFMPEG_BINDING_PATH = $BindingPath
}

function Get-PeImportedDllNames {
    param(
        [Parameter(Mandatory = $true)][string] $Path
    )

    $Objdump = Get-Command "llvm-objdump.exe" -ErrorAction SilentlyContinue
    if ($null -eq $Objdump) {
        return @()
    }

    $Output = & $Objdump.Source -p $Path 2>$null
    $Names = @()
    foreach ($Line in $Output) {
        if ($Line -match "DLL Name:\s*(.+)$") {
            $Names += $Matches[1].Trim()
        }
    }
    return $Names
}

function Copy-Clang64RuntimeDependencies {
    param(
        [Parameter(Mandatory = $true)][string[]] $EntryPoints,
        [Parameter(Mandatory = $true)][string] $DestinationDir
    )

    if (-not (Test-RunningOnWindows)) {
        return
    }

    $MsysRoot = Get-MsysRoot
    $Clang64Bin = Join-Path (Join-Path $MsysRoot "clang64") "bin"
    if (-not (Test-Path -LiteralPath $Clang64Bin -PathType Container)) {
        return
    }

    New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
    $Queue = [System.Collections.Generic.Queue[string]]::new()
    $Seen = @{}
    foreach ($EntryPoint in $EntryPoints) {
        if (Test-Path -LiteralPath $EntryPoint -PathType Leaf) {
            $FullEntryPoint = [System.IO.Path]::GetFullPath($EntryPoint)
            if (-not $Seen.ContainsKey($FullEntryPoint)) {
                $Seen[$FullEntryPoint] = $true
                $Queue.Enqueue($FullEntryPoint)
            }
        }
    }

    while ($Queue.Count -gt 0) {
        $Current = $Queue.Dequeue()
        foreach ($DllName in (Get-PeImportedDllNames -Path $Current)) {
            $Source = Join-Path $Clang64Bin $DllName
            if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
                continue
            }
            Copy-Item -LiteralPath $Source -Destination $DestinationDir -Force
            $FullSource = [System.IO.Path]::GetFullPath($Source)
            if (-not $Seen.ContainsKey($FullSource)) {
                $Seen[$FullSource] = $true
                $Queue.Enqueue($FullSource)
            }
        }
    }
}

Push-Location $Root
try {
    $RunningOnWindows = Test-RunningOnWindows
    $CargoTargetArgs = @()
    if ($RunningOnWindows) {
        $CargoTargetArgs += "--target"
        $CargoTargetArgs += "x86_64-pc-windows-gnullvm"
        if ($MediaBackend -eq "rsmpeg") {
            Initialize-Clang64RsmpegBuildEnvironment
        } else {
            Initialize-Clang64BuildEnvironment
        }
    }

    if (-not $SkipCargoBuild) {
        $CargoProfileArgs = @()
        if ($Profile -eq "release") {
            $CargoProfileArgs += "--release"
        }
        if ($MediaBackend -eq "rsmpeg") {
            cargo build @CargoProfileArgs @CargoTargetArgs -p haze
            cargo build @CargoProfileArgs @CargoTargetArgs -p haze-playout --features ffmpeg-rsmpeg
        } else {
            cargo build @CargoProfileArgs @CargoTargetArgs -p haze -p haze-playout
        }
    }

    $ProfileDir = if ($Profile -eq "release") { "release" } else { "debug" }
    $TargetProfileDir = if ($RunningOnWindows) { "target/x86_64-pc-windows-gnullvm/$ProfileDir" } else { "target/$ProfileDir" }
    $ExePath = Join-Path $Root "$TargetProfileDir/haze.exe"
    $PlayoutExePath = Join-Path $Root "$TargetProfileDir/haze-playout-rs.exe"
    if (-not (Test-Path -LiteralPath $ExePath)) {
        throw "Missing Haze executable: $ExePath"
    }
    if (-not (Test-Path -LiteralPath $PlayoutExePath)) {
        throw "Missing Rust playout executable: $PlayoutExePath"
    }

    New-Item -ItemType Directory -Force -Path $OutFull | Out-Null
    New-Item -ItemType Directory -Force -Path $BinFull | Out-Null

    $BundleOwnedFiles = @(
        "haze.exe",
        "haze.cmd",
        "README-runtime.txt",
        "config.yaml",
        ".haze-runtime"
    )
    $BundleOwnedBinFiles = @(
        "haze-web.exe",
        "haze-data-ingest.exe",
        "haze-cap-ingest.exe",
        "haze-tts.exe",
        "haze-product-render.exe",
        "haze-playlist.exe",
        "haze-ivr.exe",
        "haze-playout.exe",
        "haze-playout-rs.exe",
        "avcodec-62.dll",
        "avdevice-62.dll",
        "avfilter-11.dll",
        "avformat-62.dll",
        "avutil-60.dll",
        "libopus-0.dll",
        "libopusfile-0.dll",
        "libogg-0.dll",
        "opus.dll",
        "swresample-6.dll",
        "swscale-9.dll"
    )
    foreach ($File in $BundleOwnedFiles) {
        $Target = Join-Path $OutFull $File
        if (Test-Path -LiteralPath $Target) {
            Remove-Item -LiteralPath $Target -Force
        }
    }
    foreach ($File in $BundleOwnedBinFiles) {
        $Target = Join-Path $BinFull $File
        if (Test-Path -LiteralPath $Target) {
            Remove-Item -LiteralPath $Target -Force
        }
        $LegacyTarget = Join-Path $OutFull $File
        if (Test-Path -LiteralPath $LegacyTarget) {
            Remove-Item -LiteralPath $LegacyTarget -Force
        }
    }
    foreach ($DllDir in @($OutFull, $BinFull)) {
        if (Test-Path -LiteralPath $DllDir -PathType Container) {
            Get-ChildItem -LiteralPath $DllDir -Filter "*.dll" -File -ErrorAction SilentlyContinue |
                Remove-Item -Force
        }
    }
    foreach ($Dir in @("webroot", "managed", "audio")) {
        $Target = Join-Path $OutFull $Dir
        if (Test-Path -LiteralPath $Target) {
            Remove-Item -LiteralPath $Target -Recurse -Force
        }
    }

    Copy-Item -LiteralPath $ExePath -Destination (Join-Path $OutFull "haze.exe") -Force
    Copy-Item -LiteralPath $PlayoutExePath -Destination (Join-Path $BinFull "haze-playout-rs.exe") -Force

    if (-not $SkipGoServices) {
        & (Join-Path $ScriptDir "build-go-services.ps1") -OutputDir $OutputDir
    }
    if ($RunningOnWindows) {
        Copy-Clang64RuntimeDependencies -EntryPoints @((Join-Path $OutFull "haze.exe")) -DestinationDir $OutFull
        $ServiceEntryPoints = @()
        if (Test-Path -LiteralPath $BinFull -PathType Container) {
            $ServiceEntryPoints = @(Get-ChildItem -LiteralPath $BinFull -Filter "*.exe" -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
        }
        if ($ServiceEntryPoints.Count -gt 0) {
            Copy-Clang64RuntimeDependencies -EntryPoints $ServiceEntryPoints -DestinationDir $BinFull
        }
    }

    foreach ($Item in @("config.yaml")) {
        if (Test-Path -LiteralPath $Item) {
            Copy-Item -LiteralPath $Item -Destination $OutFull -Force
        }
    }

    foreach ($BundledDir in @("webroot", "managed", "audio")) {
        Copy-BundleDirectory -Name $BundledDir
    }
    $ManagedOut = Join-Path $OutFull "managed"
    New-Item -ItemType Directory -Force -Path $ManagedOut | Out-Null
    if ((Test-Path -LiteralPath "scripts/tts/piper_worker.py") -or (Test-Path -LiteralPath "scripts/tts/chatterbox_infer.py") -or (Test-Path -LiteralPath "scripts/tts/f5_infer.py")) {
        $ManagedScripts = Join-Path $ManagedOut "scripts"
        New-Item -ItemType Directory -Force -Path $ManagedScripts | Out-Null
        foreach ($Script in @("scripts/tts/piper_worker.py", "scripts/tts/chatterbox_infer.py", "scripts/tts/f5_infer.py")) {
            if (Test-Path -LiteralPath $Script) {
                Copy-Item -LiteralPath $Script -Destination $ManagedScripts -Force
            }
        }
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $OutFull "audio") | Out-Null

    foreach ($Dir in @("audio/_uploads", "audio/_previews", "bin", "logs", "runtime", "runtime/audio/alerts", "runtime/audio/playlist", "runtime/audio/playout", "runtime/audio/tts", "runtime/feeds", "runtime/playlists", "runtime/queues/alerts", "runtime/state")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $OutFull $Dir) | Out-Null
    }

    "Haze Weather Radio runtime directory" |
        Set-Content -LiteralPath (Join-Path $OutFull ".haze-runtime") -Encoding ASCII

    if ($IncludeEnv -and (Test-Path -LiteralPath ".env")) {
        Copy-Item -LiteralPath ".env" -Destination $OutFull -Force
    }

@"
@echo off
set "HAZE_HOME=%~dp0."
set "PATH=%~dp0bin;%PATH%"
"%~dp0haze.exe" %*
set "HAZE_EXIT=%ERRORLEVEL%"
if not "%HAZE_EXIT%"=="0" (
    echo.
    echo Haze exited with code %HAZE_EXIT%.
    echo The console is being kept open so the error above can be read.
    pause
)
exit /b %HAZE_EXIT%
"@ | Set-Content -LiteralPath (Join-Path $OutFull "haze.cmd") -Encoding ASCII

    @"
Haze Weather Radio host bundle

Run:
  haze.exe --config config.yaml

Bundled service executables are kept in:
  bin/

The daemon is the only top-level executable. Managed services are launched by
Haze and can also be extracted from its embedded service payload.

This bundle was built from:
  $Root
"@ | Set-Content -LiteralPath (Join-Path $OutFull "README-runtime.txt") -Encoding UTF8

    Write-Host "Built $OutFull"
    Write-Host "Run: $OutFull\haze.exe --config config.yaml"
} finally {
    Pop-Location
}
