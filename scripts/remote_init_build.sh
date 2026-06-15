#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"

ensure_command ssh
ensure_command git

HOST="${1:-}"
LABEL="${2:-default}"
REMOTE_BRANCH="${3:-$(git -C "$WORKSPACE_ROOT" branch --show-current)}"

if [[ -z "$HOST" ]]; then
  echo "Usage: bash scripts/remote_init_build.sh <user@host> [label] [branch]" >&2
  exit 1
fi

REMOTE_SCRIPT=$(cat <<'EOS'
set -euo pipefail
repo=""
for cand in "$HOME/research/oopsla26" "$HOME/research/oopsla"; do
  if [[ -d "$cand/.git" ]]; then
    repo="$cand"
    break
  fi
done
if [[ -z "$repo" ]]; then
  echo "ERROR: could not find remote repository under ~/research/oopsla26 or ~/research/oopsla" >&2
  exit 1
fi
cd "$repo"
mkdir -p build/remote-logs
job_label="__JOB_LABEL__"
branch="__REMOTE_BRANCH__"
mlir_branch="release/22.x"
logfile="build/remote-logs/${job_label}-driver.log"
nohup bash -lc '
set -euo pipefail
cd "'"$repo"'"
echo "[remote] repo=$PWD"
git pull --rebase origin "'"$branch"'"
git submodule sync --recursive
git submodule update --init --recursive
if [[ ! -d llvm-project/.git ]]; then
  git clone --depth 1 --branch '"$mlir_branch"' https://github.com/llvm/llvm-project.git llvm-project
fi
mkdir -p build/remote-logs
bash scripts/build_llvm_mlir.sh > "build/remote-logs/'"${job_label}"'-mlir-build.log" 2>&1
make -C choreo release > "build/remote-logs/'"${job_label}"'-choreo-build.log" 2>&1
' > "$logfile" 2>&1 < /dev/null &
pid=$!
echo "REMOTE_REPO=$repo"
echo "REMOTE_PID=$pid"
echo "REMOTE_LOG=$repo/$logfile"
EOS
)
REMOTE_SCRIPT="${REMOTE_SCRIPT//__JOB_LABEL__/$LABEL}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__REMOTE_BRANCH__/$REMOTE_BRANCH}"

ssh "$HOST" "$REMOTE_SCRIPT"
