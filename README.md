# Haze Weather Radio

Haze Weather Radio builds weather, forecast, and alert audio feeds from ECCC, NWS, and other configured sources. It includes web control, TTS, routine playout, CAP alert ingest, IVR, Icecast, and local or network audio outputs.

## Safety

Haze can generate valid SAME headers and attention tones. Only use RF or alert output where you are authorized to do so. Do not interfere with official alerting or broadcast services.

See [Hardened Accounts Operations](docs/accounts-security.md) before enabling account mode on a live operator panel.

## Build

The build scripts create a portable bundle in `dist/` containing Haze, its managed services, web files, and default configuration.

### Windows

Install Git, Rust stable, Go 1.25, FFmpeg, and the GStreamer development runtime. From PowerShell:

```powershell
git clone https://github.com/meowraii/haze-weather-radio.git
cd haze-weather-radio
.\scripts\build-haze.ps1
```

### Linux

Install the native dependencies for your platform.

#### Debian or Ubuntu

```bash
sudo apt update
sudo apt install -y \
  build-essential clang cmake pkg-config git curl \
  ffmpeg openssl redis-server libasound2-dev libopus-dev libopusfile-dev libssl-dev libudev-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

#### Fedora, Rocky Linux, or other DNF systems

```bash
sudo dnf install -y \
  gcc gcc-c++ clang cmake pkgconf-pkg-config git curl \
  ffmpeg-free ffmpeg-free-devel alsa-lib-devel openssl openssl-devel redis systemd-devel \
  opus-devel opusfile-devel gstreamer1-devel gstreamer1-plugins-base-devel \
  gstreamer1-plugins-base gstreamer1-plugins-good \
  gstreamer1-plugins-bad-free gstreamer1-plugins-ugly gstreamer1-libav
```

Fedora's `ffmpeg-free` build has restricted codec support. Enable RPM Fusion and install its `ffmpeg` packages if your required media format is unavailable. Rocky Linux may require EPEL and a multimedia repository for the equivalent FFmpeg and GStreamer plugin packages.

#### Arch Linux

```bash
sudo pacman -S --needed \
  base-devel clang cmake pkgconf git curl ffmpeg openssl redis systemd alsa-lib \
  opus opusfile gstreamer gst-plugins-base gst-plugins-good \
  gst-plugins-bad gst-plugins-ugly gst-libav
```

#### FreeBSD

```sh
sudo pkg install -y \
  bash git curl cmake pkgconf llvm gmake ffmpeg openssl redis alsa-lib opus opusfile \
  gstreamer1 gstreamer1-plugins-all
```

FreeBSD builds use the same `scripts/build-haze.sh` command. The portable bundle name is automatically tagged `FreeBSD`.

Install Rust stable and Go 1.25 or newer. The standard Rust installer works on Linux and FreeBSD:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
```

Install Go using your distribution or [go.dev](https://go.dev/dl/), then ensure `go version` reports 1.25 or newer.

Build the bundle:

```bash
git clone https://github.com/meowraii/haze-weather-radio.git
cd haze-weather-radio
bash scripts/build-haze.sh
```

## Run

Copy the completed portable bundle to its runtime machine. Keep its `managed`, `audio`, and `webroot` directories beside the Haze executable.

```bash
./haze --config config.yaml --workdir /path/to/haze-runtime
```

On Windows:

```powershell
.\haze.exe --config config.yaml --workdir C:\path\to\haze-runtime
```

Use `config.yaml` for service settings. Feed, output, voice, and product wording belong in `managed/configs/`. Use the scripts in `scripts/` for packaging and service installation.

### Public WebRTC behind NAT

Set `services.rust.media.webrtc.public_ip` to the gateway's public IPv4 address and configure a bounded `udp_port_min` and `udp_port_max`. Forward that same UDP range one-to-one from the gateway to the Haze host and allow it through the host firewall. Each concurrent WebRTC listener uses one UDP port. Environment variables `HAZE_MEDIA_WEBRTC_HOST`, `HAZE_MEDIA_WEBRTC_UDP_PORT_MIN`, and `HAZE_MEDIA_WEBRTC_UDP_PORT_MAX` override these values for machine-specific deployments.

This static ICE path works for clients that can send outbound UDP. A TURN service with short-lived credentials is still required for networks that block WebRTC UDP completely. Do not publish a permanent TURN password in `config.yaml`, panel state, logs, or the web bundle.

### Hardened account mode dependencies

Production account mode additionally requires:

- Redis running on a private or loopback interface for session leases, revocation, and atomic rate limits.
- Five private runtime variables: `HAZE_REDIS_URL`, `HAZE_PASETO_V4_LOCAL_KEY`, `HAZE_PASSWORD_PEPPER`, `HAZE_MFA_ENCRYPTION_KEY`, and `HAZE_AUDIT_HMAC_KEY`.
- HTTPS for the operator panel. Account cookies are always `Secure`, `HttpOnly`, and `SameSite=Strict`.
- Accurate system time for TOTP MFA.

Account mode fails closed if Redis or a required key is unavailable. Generate each cryptographic key independently, keep the live `.env` mode at `0600`, and never publish the environment file, account database, or audit keys through the web panel or Samba. See [Hardened Accounts Operations](docs/accounts-security.md) for Redis setup, key generation, bootstrap administration, recovery, proxy trust, and audit-log guidance.

## Project layout

- `crates/`: Rust host, media, playout, ingest, and CGEN services.
- `services/go/`: web panel, TTS, product renderer, playlist, IVR, and webhooks.
- `bundle/`: portable runtime assets and operator-managed defaults.
- `scripts/`: build, packaging, and service helpers.
