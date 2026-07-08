#!/usr/bin/env bash
# usage: nohup ./run_queue.sh queue.txt > queue.log 2>&1 &
# Queue files are text, checked into the repo per batch (reproducibility record).
# Run inside tmux; a failed config doesn't kill the night.
while IFS= read -r cmd; do
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  echo "=== $(date -Is) START: $cmd"
  eval "$cmd" || echo "=== FAILED (continuing): $cmd"
done < "$1"
