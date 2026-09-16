#!/usr/bin/env bash
# Wrapper around a training script (default: scripts/train.sh): run it in the
# background and/or tail its log live.
#
# Usage:
#   scripts/run_train.sh                          # run train.sh in the foreground
#   scripts/run_train.sh --bg                      # launch in the background, print PID + log path, return immediately
#   scripts/run_train.sh --watch                   # tail the log of the currently running background job (if any)
#   scripts/run_train.sh --bg --watch              # launch in the background, then immediately tail its log
#                                                   # (Ctrl-C only stops watching -- the training keeps running)
#   scripts/run_train.sh --script train_v1.sh --bg # same, but wrapping a different script under scripts/
#
# --script names a file under scripts/ (default: train.sh). The PID file and
# log names are derived from it (e.g. train_v1.sh -> logs/train_v1.pid,
# logs/train_v1.latest.log), so a train.sh run and a train_v1.sh run can be
# tracked -- and backgrounded -- independently without colliding.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"

SCRIPT="train.sh"
BG=0
WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bg) BG=1; shift ;;
    --watch) WATCH=1; shift ;;
    --script) SCRIPT="$2"; shift 2 ;;
    *) echo "unknown arg: $1 (expected --bg, --watch, and/or --script <name>)" >&2; exit 1 ;;
  esac
done

SCRIPT_PATH="scripts/$SCRIPT"
if [[ ! -f "$REPO_ROOT/$SCRIPT_PATH" ]]; then
  echo "no such script: $SCRIPT_PATH" >&2
  exit 1
fi
TAG="${SCRIPT%.sh}"
PID_FILE="$LOG_DIR/$TAG.pid"
LATEST_LINK="$LOG_DIR/$TAG.latest.log"

mkdir -p "$LOG_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

run_watch() {
  local log="$1"
  echo "Watching $log (Ctrl-C stops watching only -- training keeps running in the background)"
  tail -n +1 -f "$log"
}

if [[ "$BG" -eq 1 ]]; then
  if is_running; then
    echo "$SCRIPT_PATH already running (PID $(cat "$PID_FILE")). Not starting a new run." >&2
    [[ "$WATCH" -eq 1 ]] && run_watch "$LATEST_LINK"
    exit 0
  fi

  LOG_FILE="$LOG_DIR/$TAG.$(date +%Y%m%d_%H%M%S).log"
  ln -sf "$(basename "$LOG_FILE")" "$LATEST_LINK"

  cd "$REPO_ROOT"
  nohup bash "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "$SCRIPT_PATH started in background: PID $(cat "$PID_FILE"), log: $LOG_FILE"

  [[ "$WATCH" -eq 1 ]] && run_watch "$LOG_FILE"
  exit 0
fi

if [[ "$WATCH" -eq 1 ]]; then
  if is_running; then
    run_watch "$LATEST_LINK"
  else
    echo "No background $SCRIPT_PATH run is currently active -- nothing to watch. Pass --bg (optionally with --watch) to start one." >&2
    exit 1
  fi
  exit 0
fi

cd "$REPO_ROOT"
bash "$SCRIPT_PATH"
