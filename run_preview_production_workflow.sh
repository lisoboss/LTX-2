#!/usr/bin/env bash
# Run fast -> production -> modify once. Re-run safely: completed stages are skipped.

set -euo pipefail

model_root=./models
fast_prompt="A red fox runs through a snowy forest, cinematic tracking shot"
modify_prompt="The same fox pauses, looks into the camera, then walks away"
duration_seconds=5
output_dir=outputs/fox-workflow
seed=42
offload=disk

mkdir -p "$output_dir"

run_command() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

if [[ -f "$output_dir/fast.done" && -s "$output_dir/preview.mp4" && -s "$output_dir/preview.pt" ]]; then
  echo "[skip] fast"
else
  rm -f "$output_dir/production.done" "$output_dir/modify.done"
  run_command uv run python preview_production.py fast \
    --model-root "$model_root" \
    --offload "$offload" \
    --prompt "$fast_prompt" \
    --duration-seconds "$duration_seconds" \
    --artifact-path "$output_dir/preview.pt" \
    --output-path "$output_dir/preview.mp4" \
    --seed "$seed"
  touch "$output_dir/fast.done"
  echo "[success] fast"
fi

if [[ -f "$output_dir/production.done" && -s "$output_dir/production.mp4" ]]; then
  echo "[skip] production"
else
  run_command uv run python preview_production.py production \
    --model-root "$model_root" \
    --offload "$offload" \
    --artifact-path "$output_dir/preview.pt" \
    --output-path "$output_dir/production.mp4"
  touch "$output_dir/production.done"
  echo "[success] production"
fi

if [[ -f "$output_dir/modify.done" && -s "$output_dir/modified.mp4" ]]; then
  echo "[skip] modify"
else
  run_command uv run python preview_production.py modify \
    --model-root "$model_root" \
    --offload "$offload" \
    --preview-video-path "$output_dir/preview.mp4" \
    --prompt "$modify_prompt" \
    --duration-seconds "$duration_seconds" \
    --output-path "$output_dir/modified.mp4" \
    --seed "$seed"
  touch "$output_dir/modify.done"
  echo "[success] modify"
fi

echo "[complete] $output_dir"
