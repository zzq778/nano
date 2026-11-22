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
# A fixed checkout-local build location. Do not honor BUILD_DIR from an
# inherited shell environment: it could point at production build/*.so.
build_dir="$repo_root/build_chat_verify"
if [ -L "$build_dir" ] || { [ -e "$build_dir" ] && [ ! -d "$build_dir" ]; }; then
  echo "Refusing unsafe build output path: $build_dir" >&2
  exit 2
fi
jobs="${JOBS:-2}"
if [ "$jobs" != 1 ] && [ "$jobs" != 2 ]; then
  echo 'JOBS must be 1 or 2 on Jetson Nano.' >&2
  exit 2
fi
args=(-S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DWITH_TENSORRT=ON)
if [ -n "${ASTRA_SDK_ROOT:-}" ]; then
  args+=("-DASTRA_SDK_ROOT=$ASTRA_SDK_ROOT")
fi
cmake "${args[@]}"
cmake --build "$build_dir" --parallel "$jobs"
echo "Build completed: $build_dir"
echo 'Do not overwrite existing build/*.so until the behavior and ABI have been verified on Nano.'
