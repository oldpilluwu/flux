#!/usr/bin/env bash
# usage: nohup ./run_queue.sh queue.txt > queue.log 2>&1 &
# Queue files are text, checked into the repo per batch (reproducibility record).
# Run inside tmux; a failed config doesn't kill the night.

# one queue at a time — two instances OOM each other on a single GPU
exec 9>/tmp/da_queue.lock
if ! flock -n 9; then
  echo "another run_queue.sh is already running (lock: /tmp/da_queue.lock); exiting"
  exit 1
fi
while IFS= read -r cmd; do
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  echo "=== $(date -Is) START: $cmd"
  eval "$cmd" || echo "=== FAILED (continuing): $cmd"
done < "$1"
