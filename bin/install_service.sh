#!/usr/bin/env bash
# Install and enable the systemd unit so the server comes up at boot.
# usage: install_service.sh [--start]
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
install -m 0644 "$HERE/systemd/freetoken-glm53.service" /etc/systemd/system/freetoken-glm53.service
systemctl daemon-reload
systemctl enable freetoken-glm53.service
echo "enabled: $(systemctl is-enabled freetoken-glm53.service)"
if [ "${1:-}" = "--start" ]; then
  systemctl restart freetoken-glm53.service
  echo "started; follow with: journalctl -fu freetoken-glm53  (ready line: 'API server is ready to serve')"
fi
