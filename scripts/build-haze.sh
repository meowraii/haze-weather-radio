#!/usr/bin/env bash
set -euo pipefail

profile="release"
output_dir=""
media_backend="rsmpeg"
include_env=0
skip_go_services=0
skip_cargo_build=0

usage() {
  cat <<'USAGE'
Usage: scripts/build-haze.sh [options]

Options:
  --profile debug|release   Cargo profile to build (default: release)
  --output-dir DIR          Bundle output directory under dist/ (default: dist/Haze_UAP-<OS>-<ARCH>-Portable)
  --media-backend NAME      Media backend: builtin|rsmpeg (default: rsmpeg)
  --include-env             Copy .env into the bundle
  --skip-go-services        Do not build or copy managed Go service binaries
  --skip-cargo-build        Reuse an existing target artifact
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile="${2:?missing profile}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?missing output dir}"
      shift 2
      ;;
    --media-backend)
      media_backend="${2:?missing media backend}"
      shift 2
      ;;
    --include-env)
      include_env=1
      shift
      ;;
    --skip-go-services)
      skip_go_services=1
      shift
      ;;
    --skip-cargo-build)
      skip_cargo_build=1
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

if [[ "$profile" != "debug" && "$profile" != "release" ]]; then
  echo "profile must be debug or release" >&2
  exit 2
fi
if [[ "$media_backend" != "builtin" && "$media_backend" != "rsmpeg" ]]; then
  echo "media backend must be builtin or rsmpeg" >&2
  exit 2
fi

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

cd "$root"

if command -v clang >/dev/null 2>&1; then
  export CC="${CC:-clang}"
fi
if command -v clang++ >/dev/null 2>&1; then
  export CXX="${CXX:-clang++}"
fi
if command -v llvm-ar >/dev/null 2>&1; then
  export AR="${AR:-llvm-ar}"
fi

copy_bundle_dir() {
  local name="$1"
  local source="$root/bundle/$name"
  local target="$out_full/$name"
  local preserve_dir=""
  if [[ ! -d "$source" ]]; then
    return 0
  fi
  if [[ "$name" == "managed" && -d "$target" ]]; then
    preserve_dir="$(mktemp -d "${TMPDIR:-/tmp}/haze-preserve-onnx.XXXXXX")"
    while IFS= read -r -d '' file; do
      local rel="${file#"$target"/}"
      mkdir -p "$(dirname -- "$preserve_dir/$rel")"
      cp -p -- "$file" "$preserve_dir/$rel"
    done < <(find "$target" \( -type f -name '*.onnx' -o -path "$target/voices/kokoro*/*" -type f \) -print0)
  fi
  rm -rf "$target"
  cp -a "$source" "$target"
  if [[ -n "$preserve_dir" ]]; then
    while IFS= read -r -d '' file; do
      local rel="${file#"$preserve_dir"/}"
      if [[ ! -e "$target/$rel" ]]; then
        mkdir -p "$(dirname -- "$target/$rel")"
        cp -p -- "$file" "$target/$rel"
      fi
    done < <(find "$preserve_dir" -type f -print0)
    rm -rf "$preserve_dir"
  fi
}

if [[ "$skip_cargo_build" -eq 0 ]]; then
  cargo_profile_args=()
  if [[ "$profile" == "release" ]]; then
    cargo_profile_args=(--release)
  fi
  if [[ "$media_backend" == "rsmpeg" ]]; then
    cargo build "${cargo_profile_args[@]}" -p haze
    cargo build "${cargo_profile_args[@]}" -p haze-playout --features ffmpeg-rsmpeg
  else
    cargo build "${cargo_profile_args[@]}" -p haze -p haze-playout
  fi
fi

profile_dir="$profile"
exe_path="$root/target/$profile_dir/haze"
playout_exe_path="$root/target/$profile_dir/haze-playout-rs"
if [[ ! -x "$exe_path" ]]; then
  echo "Missing Haze executable: $exe_path" >&2
  exit 1
fi
if [[ ! -x "$playout_exe_path" ]]; then
  echo "Missing Rust playout executable: $playout_exe_path" >&2
  exit 1
fi

mkdir -p "$out_full" "$bin_full"
rm -f \
  "$out_full/haze" \
  "$out_full/haze.sh" \
  "$out_full/README-runtime.txt" \
  "$out_full/config.yaml" \
  "$out_full/.haze-runtime"
rm -f \
  "$out_full/haze-web" \
  "$out_full/haze-data-ingest" \
  "$out_full/haze-cap-ingest" \
  "$out_full/haze-tts" \
  "$out_full/haze-product-render" \
  "$out_full/haze-playlist" \
  "$out_full/haze-webhook" \
  "$out_full/haze-ivr" \
  "$out_full/haze-playout" \
  "$out_full/haze-playout-rs" \
  "$bin_full/haze-web" \
  "$bin_full/haze-data-ingest" \
  "$bin_full/haze-cap-ingest" \
  "$bin_full/haze-tts" \
  "$bin_full/haze-product-render" \
  "$bin_full/haze-playlist" \
  "$bin_full/haze-webhook" \
  "$bin_full/haze-ivr" \
  "$bin_full/haze-playout" \
  "$bin_full/haze-playout-rs"
rm -rf "$out_full/webroot" "$out_full/audio"

cp "$exe_path" "$out_full/haze"
chmod +x "$out_full/haze"
cp "$playout_exe_path" "$bin_full/haze-playout-rs"
chmod +x "$bin_full/haze-playout-rs"

if [[ "$skip_go_services" -eq 0 ]]; then
  bash "$script_dir/build-go-services.sh" --output-dir "$output_dir"
fi

[[ -e config.yaml ]] && cp config.yaml "$out_full/"

for bundled_dir in webroot managed audio; do
  copy_bundle_dir "$bundled_dir"
done

mkdir -p "$out_full/managed"
managed_scripts="$out_full/managed/scripts"
mkdir -p "$managed_scripts"
for script in scripts/tts/chatterbox_infer.py scripts/tts/f5_infer.py; do
  if [[ -f "$root/$script" ]]; then
    cp "$root/$script" "$managed_scripts/"
  fi
done
printf '%s\n' "Haze Weather Radio runtime directory" > "$out_full/.haze-runtime"

mkdir -p "$out_full/audio"

for dir in audio/_uploads audio/_previews bin logs runtime runtime/audio/alerts runtime/audio/playlist runtime/audio/playout runtime/audio/tts runtime/feeds runtime/playlists runtime/queues/alerts runtime/state; do
  mkdir -p "$out_full/$dir"
done

if [[ "$include_env" -eq 1 && -f .env ]]; then
  cp .env "$out_full/"
fi

cat > "$out_full/haze.sh" <<'SH'
#!/usr/bin/env sh
set -eu
bundle_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$bundle_dir/haze" "$@"
SH
chmod +x "$out_full/haze.sh"

cat > "$out_full/README-runtime.txt" <<EOF
Haze Weather Radio host bundle

Run:
  ./haze --config config.yaml

Bundled managed service executables are kept in:
  bin/

This bundle was built from:
  $root
EOF

echo "Built $out_full"
echo "Run: $out_full/haze --config config.yaml"
