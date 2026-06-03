#!/bin/bash
set -e

# AWS CLI 가 moto 허니팟으로 라우팅되도록 환경변수 주입
echo "AWS_ENDPOINT_URL=http://moto:5000" >> /etc/environment
echo "AWS_DEFAULT_REGION=us-east-1" >> /etc/environment

# 세션 로그 디렉터리 권한 설정 (볼륨 마운트 후 적용)
chmod 777 /var/log/honeypot

exec /usr/sbin/sshd -D -e
