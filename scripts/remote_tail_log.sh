#!/usr/bin/env bash

set -euo pipefail

ensure_usage() {
  if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/remote_tail_log.sh <user@host> <remote-log-path> [tail-args...]" >&2
    exit 1
  fi
}

ensure_usage "$@"
HOST="$1"
shift
LOG_PATH="$1"
shift || true

ssh "$HOST" "tail ${*:- -n 50} '$LOG_PATH'"
