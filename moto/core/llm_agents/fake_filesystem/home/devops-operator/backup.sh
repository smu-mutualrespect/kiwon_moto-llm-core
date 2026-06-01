#!/bin/bash
set -euo pipefail

REGION="us-east-1"
BUCKET="nexora-backup-prod"
SOURCE_DIR="/data"
LOG_DIR="/home/devops-operator/logs"
LOG_FILE="$LOG_DIR/backup-$(date -u +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

{
  echo "$(date -u +%FT%TZ) backup start source=$SOURCE_DIR bucket=s3://$BUCKET/daily region=$REGION"
  aws s3 sync "$SOURCE_DIR" "s3://$BUCKET/daily" --region "$REGION"
  echo "$(date -u +%FT%TZ) backup complete bucket=s3://$BUCKET/daily"
} | tee -a "$LOG_FILE"
