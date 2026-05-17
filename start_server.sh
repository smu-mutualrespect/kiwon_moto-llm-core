#!/usr/bin/env bash
# AgentHoneypot moto_server 시작 스크립트
# Usage: ./start_server.sh [port]  (기본값: 5000)

set -euo pipefail

PORT="${1:-5000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
LOG_FILE="/tmp/moto_server_${PORT}.log"
PID_FILE="/tmp/moto_server_${PORT}.pid"

# 이미 실행 중이면 종료
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[*] 기존 서버(PID=$(cat "$PID_FILE"))를 종료합니다..."
    kill "$(cat "$PID_FILE")"
    sleep 1
fi

# venv 활성화
VENV="${SCRIPT_DIR}/../honeypot-env"
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

# .env 파일에서 환경 변수 로드
if [ -f "$ENV_FILE" ]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
    echo "[*] .env 로드 완료 (MOTO_LLM_PROVIDER=${MOTO_LLM_PROVIDER:-not set})"
else
    echo "[!] .env 파일 없음 — LLM 연동 없이 실행됩니다."
fi

# 백그라운드로 실행 (터미널 닫아도 유지)
nohup python3 -m moto.server -p "$PORT" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "[✓] moto_server 시작 완료 (PID=$PID, port=$PORT)"
    echo "    로그: tail -f $LOG_FILE"
    echo "    종료: kill $PID  또는  $0 stop"
else
    echo "[✗] 서버 시작 실패 — 로그 확인: $LOG_FILE"
    cat "$LOG_FILE"
    exit 1
fi
