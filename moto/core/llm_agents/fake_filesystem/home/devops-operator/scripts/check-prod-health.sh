#!/usr/bin/env bash
set -euo pipefail

aws sts get-caller-identity --region us-east-1
kubectl get deploy -n prod
kubectl get hpa -n prod
kubectl get ingress -n prod
aws cloudwatch get-metric-statistics \
  --region us-east-1 \
  --namespace AWS/ApplicationELB \
  --metric-name HTTPCode_Target_5XX_Count \
  --statistics Sum \
  --period 300 \
  --start-time 2024-05-26T07:00:00Z \
  --end-time 2024-05-26T08:00:00Z
