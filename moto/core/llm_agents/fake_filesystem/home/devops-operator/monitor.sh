#!/bin/bash
set -euo pipefail

LOG_DIR="/home/devops-operator/logs"
LOG_FILE="$LOG_DIR/monitor-$(date -u +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

{
  echo "== caller =="
  aws sts get-caller-identity
  echo "== services =="
  systemctl status nginx --no-pager
  systemctl status backend-api --no-pager
  echo "== docker =="
  docker ps
  echo "== prod pods =="
  kubectl get pods -n prod
} | tee -a "$LOG_FILE"
