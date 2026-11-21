#!/usr/bin/env bash
# Run this on your phone in Termux after installing openssh.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: scripts/ssh_jetson_check.sh user@IP-or-hostname' >&2
  exit 2
fi
command -v ssh >/dev/null || { echo 'Missing ssh. On Termux: pkg install openssh' >&2; exit 2; }
ssh -o ConnectTimeout=8 -o ServerAliveInterval=10 -- "$1" \
  'printf "arch: "; uname -m; printf "os: "; . /etc/os-release && echo "$PRETTY_NAME"; printf "kernel: "; uname -r; printf "gcc: "; g++ --version 2>/dev/null | head -1 || true; printf "cuda: "; ls -d /usr/local/cuda* 2>/dev/null || true'
