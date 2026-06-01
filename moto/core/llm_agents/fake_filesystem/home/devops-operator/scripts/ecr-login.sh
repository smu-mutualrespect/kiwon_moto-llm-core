#!/usr/bin/env bash
set -euo pipefail

REGISTRY="847362915408.dkr.ecr.us-east-1.amazonaws.com"
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "$REGISTRY"
