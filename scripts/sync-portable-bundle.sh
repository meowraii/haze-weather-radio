#!/usr/bin/env bash
set -euo pipefail

output_dir=""
reverse=0
dry_run=0

usage() {
  cat <<'USAGE'
Usage: scripts/sync-portable-bundle.sh [options]

Options:
  --output-dir DIR   Portable output directory under dist/ (default: dist/Haze_UAP-<OS>-<ARCH>-Portable)
  --reverse          Sync portable bundle files back into repo bundle/
  --dry-run          Preview changes without copying or deleting
  -h, --help         Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="${2:?missing output dir}"
      shift 2
      ;;
    --reverse)
      reverse=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
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
mkdir -p "$dist_root"
dist_full="$(cd -- "$dist_root" && pwd)"

case "$out_full" in
  "$dist_full"|"$dist_full"/*) ;;
  *)
    echo "Refusing to sync outside the dist directory: $out_full" >&2
    exit 1
    ;;
esac

if [[ ! -d "$out_full" ]]; then
  echo "Portable directory does not exist: $out_full" >&2
  exit 1
fi

if [[ "$reverse" -eq 1 ]]; then
  from_root="$out_full"
  to_root="$root/bundle"
  echo "Syncing portable bundle files back into repo bundle..."
else
  from_root="$root/bundle"
  to_root="$out_full"
  echo "Syncing repo bundle files into portable bundle..."
fi

sync_dir() {
  local name="$1"
  local src="$from_root/$name"
  local dst="$to_root/$name"
  if [[ ! -d "$src" ]]; then
    echo "Skipping missing source: $src"
    return
  fi
  echo "$name: $src -> $dst"
  if command -v rsync >/dev/null 2>&1; then
    if [[ "$dry_run" -eq 1 ]]; then
      rsync -av --delete --exclude='*.onnx' --dry-run "$src/" "$dst/"
    else
      mkdir -p "$dst"
      rsync -a --delete --exclude='*.onnx' "$src/" "$dst/"
    fi
    return
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    echo "rsync is unavailable; dry-run fallback can only report the planned mirror."
    return
  fi
  mkdir -p -- "$dst"
  cp -a -- "$src/." "$dst/"
}

for name in webroot managed audio; do
  sync_dir "$name"
done

echo "Bundle sync complete."
