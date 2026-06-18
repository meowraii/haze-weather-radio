#!/usr/bin/env bash
set -euo pipefail

output_dir=""

usage() {
  cat <<'USAGE'
Usage: scripts/build-go-services.sh [options]

Options:
  --output-dir DIR   Bundle output directory under dist/ (default: dist/Haze_UAP-<OS>-<ARCH>-Portable)
  -h, --help         Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="${2:?missing output dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/.." && pwd)"
dist_root="$root/dist"

portable_os() {
  case "$(uname -s)" in
    Linux*) echo "Linux" ;;
    FreeBSD*) echo "FreeBSD" ;;
    Darwin*) echo "macOS" ;;
    MINGW*|MSYS*|CYGWIN*) echo "Windows" ;;
    *) uname -s | tr -cd '[:alnum:]_-' ;;
  esac
}

portable_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    armv7l|armv7*) echo "armv7" ;;
    armv6l|armv6*) echo "armv6" ;;
    i386|i686) echo "x86" ;;
    *) uname -m | tr -cd '[:alnum:]_-' ;;
  esac
}

portable_output_dir() {
  printf 'dist/Haze_UAP-%s-%s-Portable\n' "$(portable_os)" "$(portable_arch)"
}

resolve_path() {
  case "$1" in
    /*)
      mkdir -p "$(dirname -- "$1")"
      printf '%s/%s\n' "$(cd -- "$(dirname -- "$1")" && pwd)" "$(basename -- "$1")"
      ;;
    *)
      mkdir -p "$root/$(dirname -- "$1")"
      printf '%s/%s\n' "$(cd -- "$root/$(dirname -- "$1")" && pwd)" "$(basename -- "$1")"
      ;;
  esac
}

if [[ -z "$output_dir" ]]; then
  output_dir="$(portable_output_dir)"
fi

out_full="$(resolve_path "$output_dir")"
bin_full="$out_full/bin"
mkdir -p "$dist_root"
dist_full="$(cd -- "$dist_root" && pwd)"

case "$out_full" in
  "$dist_full"|"$dist_full"/*) ;;
  *)
    echo "Refusing to write outside the dist directory: $out_full" >&2
    exit 1
    ;;
esac

mkdir -p "$out_full" "$bin_full" "$root/target/go-build-cache" "$root/target/go-tmp"
rm -f \
  "$out_full/haze-web" \
  "$out_full/haze-data-ingest" \
  "$out_full/haze-cap-ingest" \
  "$out_full/haze-tts" \
  "$out_full/haze-product-render" \
  "$out_full/haze-playlist" \
  "$out_full/haze-ivr"

copy_bundle_dir() {
  local name="$1"
  local source="$root/bundle/$name"
  local target="$out_full/$name"
  if [[ ! -d "$source" ]]; then
    return 0
  fi
  rm -rf "$target"
  cp -a "$source" "$target"
}

cd "$root/services/go"
export GOCACHE="$root/target/go-build-cache"
export GOTMPDIR="$root/target/go-tmp"

build_web_args=()
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists opus; then
  export CGO_ENABLED=1
  build_web_args=(-tags opus_cgo)
elif [[ "${HAZE_ALLOW_NO_OPUS:-}" != "1" && "${HAZE_ALLOW_NO_OPUS:-}" != "true" ]]; then
  echo "Native Opus build inputs are required for receiver/WebRTC audio. Install pkg-config and libopus development files, or set HAZE_ALLOW_NO_OPUS=1 for a degraded dev-only build." >&2
  exit 1
else
  echo "Warning: building haze-web without native Opus support; receiver Opus/WebRTC audio will be degraded." >&2
fi

go build "${build_web_args[@]}" -o "$bin_full/haze-web" ./cmd/haze-web
go build -o "$bin_full/haze-data-ingest" ./cmd/haze-data-ingest
go build -o "$bin_full/haze-cap-ingest" ./cmd/haze-cap-ingest
go build -o "$bin_full/haze-tts" ./cmd/haze-tts
go build -o "$bin_full/haze-product-render" ./cmd/haze-product-render
go build -o "$bin_full/haze-playlist" ./cmd/haze-playlist
go build -o "$bin_full/haze-ivr" ./cmd/haze-ivr

chmod +x \
  "$bin_full/haze-web" \
  "$bin_full/haze-data-ingest" \
  "$bin_full/haze-cap-ingest" \
  "$bin_full/haze-tts" \
  "$bin_full/haze-product-render" \
  "$bin_full/haze-playlist" \
  "$bin_full/haze-ivr"

for bundled_dir in webroot managed audio; do
  copy_bundle_dir "$bundled_dir"
done

echo "Built Go services in $bin_full"
