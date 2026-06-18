# haze-weather-radio
Haze Weather Radio is a Rust-hosted, Go-service weather radio system for generating, managing, and broadcasting weather information and emergency alert audio. The production bundle is built around `haze`, which owns startup, shutdown, logging, runtime directory selection, and bundled service supervision. Go services are taking over I/O-heavy work such as the web gateway, CAP ingest, and TTS rendering.

# Why
On March 16, 2026, Environment Canada shut down their Weatheradio Canada service, which provided weather radio broadcasts across the country. In response to this, I decided to create Haze Weather Radio as a replacement for the service, using publicly available data feeds and open-source tools. My goal is to provide a free and accessible weather radio service for Canadians, and to keep the spirit of Weatheradio Canada alive in a new and modernized form. As well as emphasize the importance of redundancy, accessibility, and reliability in public safety communications, and to demonstrate how technology can be used to fill gaps in public services when they arise.

# Over-The-Air Broadcasting Disclaimer
As Haze Weather Radio contains generation of valid SAME headers, it is intended for use in compliance with local regulations and broadcasting laws. Users are responsible for ensuring that their use of the software adheres to all applicable legal requirements. Broadcasting valid SAME headers runs the risk of interfering with official Emergency Alert System (EAS) equipment or weather alert radios and may be subject to legal penalties if used improperly. It is recommended to use the software for testing and educational purposes only, and to avoid broadcasting on frequencies that may interfere with official EAS broadcasts. Even if a certain someone says their transmitter only goes 10 feet.
I, the developer of Haze Weather Radio, disclaim any responsibility for misuse of the software or any legal consequences that may arise from its use.

# Features
- Weather data from ECCC, TWC, and NWS APIs.
- A centralized management system for multiple feeds servicing different locations.
- Go web gateway with a web interface for monitoring, alert origination, and management.
- In-app support for English, French, and Spanish, with the ability to add more languages via configuration.
- Support for multiple output methods including Icecast streaming, local audio playback, and network audio transport sinks (UDP/RTP/RTMP/SRT/RTSP).
- Integration with official CAP feeds such as NAADS (Alert Ready/NPAS) for real-time weather alerts.
- The Weather On-Demand interface allows applications to generate custom audio or text packages for specific locations and conditions on demand. (e.g. An IVR system that provides weather updates for a caller's location, or a smart home device that announces weather forecasts in the morning.)

# Haze Host
Haze uses `haze` as the primary host application. It owns process startup, shutdown, logging, runtime directory selection, and bundled service supervision.

```powershell
scripts/build-haze.ps1
dist/Haze_UAP-Windows-x86_64-Portable/haze.exe --config config.yaml
```

The production bundle is host-only and ships the `haze` executable plus bundled Go services.

On first launch, if the executable directory already contains meaningful files, `haze` asks whether to keep runtime files there, create a `haze-runtime` subfolder, or choose a custom directory. Services and unattended installs should pass `--workdir C:\path\to\runtime` to skip the prompt.

# Media Backend
Haze has a shared `haze-media` crate for PCM shape, WAV decode, normalization, and the FFmpeg/libav backend boundary. Portable builds use the built-in PCM backend by default so they can still compile without FFmpeg development libraries:

```powershell
scripts/build-haze.ps1 -MediaBackend builtin
```

The lower-level FFmpeg path is exposed through `rsmpeg` and can be selected once FFmpeg headers and libraries are available:

```powershell
$env:FFMPEG_LIBS_DIR = "C:\path\to\ffmpeg\lib"
$env:FFMPEG_INCLUDE_DIR = "C:\path\to\ffmpeg\include"
scripts/build-haze.ps1 -MediaBackend rsmpeg
```

For the local Windows MSVC build, vcpkg is the preferred FFmpeg provider:

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"
$env:VCPKGRS_DYNAMIC = "1"
scripts/build-haze.ps1 -MediaBackend rsmpeg-vcpkg
```

This is intended to replace external `ffmpeg` subprocess encoder paths with one unified in-process media layer over time.

# Go Service Migration
The service migration target is Go for I/O-heavy Haze services. First-pass services live under `services/go`:

```powershell
scripts/build-go-services.ps1
dist/Haze_UAP-Windows-x86_64-Portable/bin/haze-web.exe --addr 127.0.0.1:8081 --webroot webroot --config config.yaml
dist/Haze_UAP-Windows-x86_64-Portable/bin/haze-cap-ingest.exe --source naads --once
```

`haze-web` is a static/admin/public gateway scaffold with health and WebSocket endpoints. `haze-cap-ingest` is intentionally policy-free: it fetches Atom/CAP sources and emits normalized `cap.alert.received` JSON events into the Haze host bridge when launched by `haze`, or JSONL to stdout when run standalone. `haze-tts` runs as a host-bridge TTS renderer for SAPI5, eSpeak, and Piper readers.

The transmitter-side receiver stays separate as `hazeReceiver.py`; it pairs with the admin server, receives transmitter parameters, and plays the secure WebRTC feed into `pi_fm_adv` on the target host.

# Host Processing
The `haze` host includes a first-pass SAME header/AFSK/tone generation component under `crates/haze/src/same_core.rs`. Deterministic signal generation belongs in the host as the remaining playout and scheduling pieces are migrated.

# Planned Features
- Make the web interface not look vibecoded by Claude. **too damn lazy. if you hate vibecoded frontends, feel free to open a PR. though i did get banned from Claude if that makes you feel better.**
- Be better than Weatheradio Canada.
- **Support for additional CAP feeds such as those from the US NWS (NWS-CAP ATOM, IPAWSOPEN, etc.), and international sources. (NWS-CAP has been implemented, but not thoroughly tested yet. Because I am Canadian.)**
