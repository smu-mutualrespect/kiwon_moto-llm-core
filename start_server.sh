#!/usr/bin/env bash
# AgentHoneypot moto_server 시작 스크립트
# Usage: ./start_server.sh [port] | ./start_server.sh stop [port]  (기본값: 5000)

set -euo pipefail

COMMAND="${1:-start}"
if [ "$COMMAND" = "stop" ]; then
    PORT="${2:-5000}"
else
    PORT="${1:-5000}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

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

SERVER_PYTHON="${MOTO_SERVER_PYTHON:-python3}"

DATA_ROOT="${MOTO_HONEYPOT_DATA_ROOT:-/tmp/phantomgate}"
PID_FILE="${MOTO_HONEYPOT_PID_FILE:-${DATA_ROOT}/moto_server_${PORT}.pid}"

ensure_dir() {
    local path="$1"
    local label="$2"

    if ! mkdir -p "$path" 2>/dev/null; then
        cat >&2 <<EOF
[✗] ${label} 디렉터리를 만들 수 없습니다: ${path}
    MOTO_HONEYPOT_DATA_ROOT 또는 MOTO_HONEYPOT_RUN_ROOT를 쓰기 가능한 경로로 지정하세요.
EOF
        exit 1
    fi
}

ensure_dir "$DATA_ROOT" "데이터 루트"
mkdir -p "$(dirname "$PID_FILE")"

if [ "$COMMAND" = "stop" ]; then
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[*] 서버(PID=$(cat "$PID_FILE"), port=$PORT)를 종료합니다..."
        kill "$(cat "$PID_FILE")"
        rm -f "$PID_FILE"
        exit 0
    fi
    echo "[*] 실행 중인 서버가 없습니다. (port=$PORT)"
    exit 0
fi

RUN_ID="${MOTO_HONEYPOT_RUN_ID:-${PORT}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${MOTO_HONEYPOT_RUN_ROOT:-${DATA_ROOT}/runs/${RUN_ID}}"
REPORT_DIR="${MOTO_HONEYPOT_REPORT_DIR:-${RUN_ROOT}/reports}"
LOG_FILE="${MOTO_HONEYPOT_LOG_FILE:-${RUN_ROOT}/server.log}"

ensure_dir "$RUN_ROOT" "실행 루트"
ensure_dir "$REPORT_DIR" "보고서"
mkdir -p "$(dirname "$LOG_FILE")"

export MOTO_HONEYPOT_REPORT_DIR="$REPORT_DIR"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if ! "$SERVER_PYTHON" -c "import boto3" >/dev/null 2>&1; then
    cat >&2 <<EOF
[✗] ${SERVER_PYTHON} 환경에 boto3가 없습니다.
    서버에 사용할 Python/venv에 서버 의존성을 설치하거나 MOTO_SERVER_PYTHON을 지정하세요.
    이 스크립트는 PYTHONPATH로 현재 저장소 코드를 읽으므로 editable install은 필요 없습니다.
    예:
      python3 -m venv ${DATA_ROOT}/venv
      ${DATA_ROOT}/venv/bin/pip install boto3 'botocore!=1.35.45,!=1.35.46' cryptography requests xmltodict werkzeug python-dateutil responses Jinja2 flask
      MOTO_SERVER_PYTHON=${DATA_ROOT}/venv/bin/python $0 ${PORT}
EOF
    exit 1
fi

# 이미 실행 중이면 종료
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[*] 기존 서버(PID=$(cat "$PID_FILE"))를 종료합니다..."
    kill "$(cat "$PID_FILE")"
    sleep 1
fi

# 백그라운드로 실행 (터미널 닫아도 유지)
cd "$RUN_ROOT"
nohup "$SERVER_PYTHON" -m moto.server -p "$PORT" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "[✓] moto_server 시작 완료 (PID=$PID, port=$PORT)"
    echo "    실행 루트: $RUN_ROOT"
    echo "    보고서: $REPORT_DIR"
    echo "    로그: tail -f $LOG_FILE"
    echo "    종료: kill $PID  또는  $0 stop $PORT"
else
    echo "[✗] 서버 시작 실패 — 로그 확인: $LOG_FILE"
    cat "$LOG_FILE"
    exit 1
fi
