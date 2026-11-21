#!/usr/bin/env bash
# CPU-only NMS equivalence check. Runs on a Linux ARM64 userspace (e.g. Ubuntu in proot) or Jetson.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
command -v g++ >/dev/null || { echo 'Missing g++. On Ubuntu: apt install g++' >&2; exit 2; }
out="${TMPDIR:-/tmp}/jetson_vision_nms_${USER:-user}"
g++ -std=c++14 -O2 -Iinclude src/detection_postprocessor.cpp tests/nms_equivalence.cpp -o "$out"
"$out"
if command -v file >/dev/null; then file "$out"; fi
