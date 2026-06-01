#!/bin/bash
set -euo pipefail

LOG_DIR="/home/devops-operator/logs"
LOG_FILE="$LOG_DIR/restart-api-$(date -u +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

{
  echo "$(date -u +%FT%TZ) restart start service=nginx,backend-api user=devops-operator"
  systemctl restart nginx
  systemctl restart backend-api
  systemctl status nginx --no-pager
  systemctl status backend-api --no-pager
  echo "$(date -u +%FT%TZ) restart complete service=nginx,backend-api"
} | tee -a "$LOG_FILE"
