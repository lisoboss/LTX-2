#!/usr/bin/env bash
# Run fast -> production -> enhance once. Re-run safely: completed stages are skipped.

set -euo pipefail

model_root=./models
fast_prompt="A red fox runs through a snowy forest, cinematic tracking shot"
enhance_prompt="The same fox pauses, looks into the camera, then walks away"
duration_seconds=5
output_dir=outputs/fox-workflow
seed=42
offload=disk
workflow_version=2
enhance_version=1

uv_bin=${UV_BIN:-uv}
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ ! -x "$uv_bin" ]]; then
  echo "uv not found. Set UV_BIN or install uv at $HOME/.local/bin/uv." >&2
  exit 1
fi

mkdir -p "$output_dir"

timestamp_log() {
  while IFS= read -r line; do
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$line"
  done
}

exec > >(timestamp_log | tee -a "$output_dir/workflow.log") 2>&1
echo "===== workflow started ====="

run_command() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

if [[ -f "$output_dir/fast.done" && "$(<"$output_dir/fast.done")" == "$workflow_version" && -s "$output_dir/preview.mp4" && -s "$output_dir/preview.pt" ]]; then
  echo "[skip] fast"
else
  rm -f "$output_dir/production.done" "$output_dir/enhance.done"
  run_command "$uv_bin" run python preview_production.py fast \
    --model-root "$model_root" \
    --offload "$offload" \
    --prompt "$fast_prompt" \
    --duration-seconds "$duration_seconds" \
    --artifact-path "$output_dir/preview.pt" \
    --output-path "$output_dir/preview.mp4" \
    --seed "$seed"
  printf '%s\n' "$workflow_version" > "$output_dir/fast.done"
  echo "[success] fast"
fi

if [[ -f "$output_dir/production.done" && "$(<"$output_dir/production.done")" == "$workflow_version" && -s "$output_dir/production.mp4" ]]; then
  echo "[skip] production"
else
  run_command "$uv_bin" run python preview_production.py production \
    --model-root "$model_root" \
    --offload "$offload" \
    --artifact-path "$output_dir/preview.pt" \
    --output-path "$output_dir/production.mp4"
  printf '%s\n' "$workflow_version" > "$output_dir/production.done"
  echo "[success] production"
fi

if [[ -f "$output_dir/enhance.done" && "$(<"$output_dir/enhance.done")" == "$enhance_version" && -s "$output_dir/enhanced.mp4" ]]; then
  echo "[skip] enhance"
else
  run_command "$uv_bin" run python preview_production.py enhance \
    --model-root "$model_root" \
    --offload "$offload" \
    --production-video-path "$output_dir/production.mp4" \
    --prompt "$enhance_prompt" \
    --duration-seconds "$duration_seconds" \
    --output-path "$output_dir/enhanced.mp4" \
    --seed "$seed"
  printf '%s\n' "$enhance_version" > "$output_dir/enhance.done"
  echo "[success] enhance"
fi

echo "[complete] $output_dir"
