#!/usr/bin/env bash
# Optional opt-in boot service. Does not run as root and does not manage production vision.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(id -u)" -eq 0 ]]; then
    echo "Run as the normal Jetson login user (e.g. rikibot), not with sudo." >&2
    exit 2
fi
user_name="$(id -un)"
python_path="$(command -v python3)"
service_name=jetson-chat-bridge.service

if [[ ! "$user_name" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo "Unexpected username; refusing to generate a service" >&2
    exit 2
fi

if [[ ! -f "$repo_root/tools/remote_bridge.py" ]]; then
    echo "Bridge program not found" >&2
    exit 2
fi

unit="$(mktemp)"
trap 'rm -f "$unit"' EXIT
cat > "$unit" <<EOF
[Unit]
Description=Jetson ChatGPT GitHub development bridge
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$user_name
WorkingDirectory=$repo_root
ExecStart=$python_path -u $repo_root/tools/remote_bridge.py loop
Restart=on-failure
RestartSec=20
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

if [[ "${1:---preview}" != "--install" ]]; then
    echo "Preview only. No system changes have been made."
    cat "$unit"
    echo
    echo "After checking this file, run: bash scripts/install_bridge_service.sh --install"
    exit 0
fi

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--preview|--install]" >&2
    exit 2
fi

# The script stops only this user's exact prior background agent, then installs
# a system service that runs *as the same unprivileged user*.
if pgrep -f '^python3 -u tools/remote_bridge.py loop$' >/dev/null; then
    echo "Stopping previous user-owned nohup bridge before switching to systemd"
    pgrep -f '^python3 -u tools/remote_bridge.py loop$' | xargs -r kill
fi
sudo install -m 0644 "$unit" "/etc/systemd/system/$service_name"
sudo systemctl daemon-reload
sudo systemctl enable --now "$service_name"
sudo systemctl --no-pager --full status "$service_name"