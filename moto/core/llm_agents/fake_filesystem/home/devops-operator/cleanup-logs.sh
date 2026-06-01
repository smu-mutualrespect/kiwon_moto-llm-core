#!/bin/bash
set -euo pipefail

LOG_DIR="/home/devops-operator/logs"
LOG_FILE="$LOG_DIR/cleanup-logs-$(date -u +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

{
  echo "$(date -u +%FT%TZ) cleanup start"
  find /home/devops-operator/logs -type f -name "*.log" -mtime +14 -print -delete
  find /var/log/nginx -type f -name "*.log.*" -mtime +14 -print -delete
  journalctl --vacuum-time=7d
  echo "$(date -u +%FT%TZ) cleanup complete"
} | tee -a "$LOG_FILE"
