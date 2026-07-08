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

# pre-flight: refuse to start while the GPU is still occupied — leftover jobs
# from older runners (started before the lock existed) and slow CUDA teardown
# both present as a mystery OOM at job 1. FORCE=1 to override.
used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ -z "${FORCE:-}" && "${used_mib:-0}" -gt 2048 ]]; then
  echo "GPU already has ${used_mib} MiB in use:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  echo "kill the processes above first (or FORCE=1 to override); exiting"
  exit 1
fi

while IFS= read -r cmd; do
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  echo "=== $(date -Is) START: $cmd"
  eval "$cmd" || echo "=== FAILED (continuing): $cmd"
done < "$1"
