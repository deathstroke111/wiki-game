#!/usr/bin/env bash
# SessionStart hook: detect which machine this session is running on and tell
# Claude via additionalContext. No jq dependency (rented box may not have it).
#   - Darwin + Apple Silicon        -> local dev machine (MacBook M4, 24 GB RAM)
#   - Linux + working nvidia-smi    -> rented CUDA machine (RTX, 16 GB VRAM, 64 GB RAM)
#   - anything else                 -> unknown, flag it

os="$(uname -s)"
arch="$(uname -m)"
host="$(hostname 2>/dev/null || echo unknown)"

if [ "$os" = "Darwin" ]; then
  chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
  machine="LOCAL_DEV"
  msg="Machine: LOCAL_DEV (macOS ${arch}, ${chip}, ${mem_gb} GB unified RAM, host ${host}). No CUDA: use device 'mps' or 'cpu'. Use for development, unit tests, tiny smoke runs only; no real training. Use the project .venv via uv."
elif [ "$os" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1)"
  mem_gb="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null)"
  machine="CUDA_RENTED"
  msg="Machine: CUDA_RENTED (Linux ${arch}, GPU: ${gpu}, ${mem_gb} GB system RAM, host ${host}). Rented and not always available: real training and large evals run here, device 'cuda'. Use the project .venv via uv, and checkpoint/log often since the box may disappear."
else
  machine="UNKNOWN"
  msg="Machine: UNKNOWN (${os} ${arch}, host ${host}). Matches neither the local M4 MacBook nor the rented CUDA box (no working nvidia-smi on Linux). Ask the user before running anything heavy."
fi

# Escape for JSON (msg contains no quotes/backslashes by construction, but be safe)
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

printf '{"systemMessage":"%s","hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' \
  "$(esc "[$machine] $host")" "$(esc "$msg")"
