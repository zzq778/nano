#!/usr/bin/env bash
# Builds in a separate directory; does not replace production .so files.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$(uname -m)" != 'aarch64' ]; then
  echo 'This build requires a Jetson ARM64 host (uname -m = aarch64).' >&2
  exit 2
fi
command -v cmake >/dev/null || { echo 'cmake is missing' >&2; exit 2; }
command -v g++ >/dev/null || { echo 'g++ is missing' >&2; exit 2; }
# Build dir is intentionally distinct from build/ used by the production app.
build_dir="${BUILD_DIR:-build_phone_verify}"
args=(-S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DWITH_TENSORRT=ON)
if [ -n "${ASTRA_SDK_ROOT:-}" ]; then
  args+=("-DASTRA_SDK_ROOT=$ASTRA_SDK_ROOT")
fi
cmake "${args[@]}"
cmake --build "$build_dir" --parallel "${JOBS:-2}"
echo "Build completed: $build_dir"
echo 'Do not overwrite existing build/*.so until the behavior and ABI have been verified on Nano.'
