#!/bin/bash
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="847362915408"
REGISTRY="847362915408.dkr.ecr.us-east-1.amazonaws.com"
IMAGE="847362915408.dkr.ecr.us-east-1.amazonaws.com/nexora-backend-api:latest"
SERVICE="backend-api"
LOG_DIR="/home/devops-operator/logs"
LOG_FILE="$LOG_DIR/deploy-$(date -u +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

{
  echo "$(date -u +%FT%TZ) deploy start user=devops-operator account=$ACCOUNT_ID region=$REGION service=$SERVICE"
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"
  docker pull "$IMAGE"
  docker-compose up -d "$SERVICE"
  docker-compose ps "$SERVICE"
  echo "$(date -u +%FT%TZ) deploy complete image=$IMAGE"
} | tee -a "$LOG_FILE"
