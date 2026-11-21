#!/usr/bin/env bash
# One-time upgrade of this user's already-configured bridge; no sudo required.
set -euo pipefail
if [[ "$(id -u)" -eq 0 ]]; then
    echo 'Do not run the bridge as root; use the normal Jetson login.' >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Fail on user changes rather than silently clobbering them.
git pull --ff-only
python3 -m pytest -q tests/test_remote_bridge.py tests/test_dev_agent.py

if systemctl is-active --quiet jetson-chat-bridge.service 2>/dev/null; then
    echo 'systemd service detected; restart via: sudo systemctl restart jetson-chat-bridge.service'
    exit 0
fi

# Replace only the exact old bridge command line owned by this user.
pids="$(pgrep -f '^python3 -u tools/remote_bridge.py loop$' || true)"
if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill
fi
nohup python3 -u tools/remote_bridge.py loop > "$HOME/jetson-bridge.log" 2>&1 < /dev/null &
echo "Bridge v2 started with PID $!."
sleep 2
if ! kill -0 "$!" 2>/dev/null; then
    echo 'Bridge process exited; check ~/jetson-bridge.log' >&2
    tail -n 25 "$HOME/jetson-bridge.log" >&2
    exit 1
fi
tail -n 10 "$HOME/jetson-bridge.log"