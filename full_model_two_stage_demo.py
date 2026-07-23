"""Generate a full LTX-2.3 22B two-stage production render for fast-preview comparison.

The existing ``full-model-fast.mp4`` is a decoded video, while Stage 2 consumes
Stage-1 latents rather than pixels.  This standalone demo therefore reruns Stage 1
with the same prompt and seed, then performs native latent upsampling and Stage 2.
It changes no package source.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio, SpatioTemporalScaleFactors
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode

MODEL_ROOT = Path("./models")
CHECKPOINT_RELATIVE_PATH = Path("LTX-2.3/ltx-2.3-22b-dev.safetensors")
DISTILLED_LORA_RELATIVE_PATH = Path("LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
SPATIAL_UPSAMPLER_RELATIVE_PATH = Path("LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
GEMMA_RELATIVE_PATH = Path("gemma-3-12b-it-qat-q4_0-unquantized")
HEIGHT = 768
WIDTH = 1280
FRAME_RATE = 24.0
DEFAULT_STAGE_1_STEPS = 8
DEFAULT_OFFLOAD = OffloadMode.DISK


def duration_to_num_frames(duration_seconds: float) -> int:
    """Return the largest LTX-compatible frame count within the requested duration."""
    if not torch.isfinite(torch.tensor(duration_seconds)) or duration_seconds <= 0:
        raise ValueError("duration-seconds must be a finite positive number")
    requested_frames = max(1, round(duration_seconds * FRAME_RATE))
    time_scale = SpatioTemporalScaleFactors.default().time
    frames = ((requested_frames - 1) // time_scale) * time_scale + 1
    if frames < time_scale + 1:
        raise ValueError(f"duration-seconds must produce at least {time_scale + 1} frames")
    return frames


def _inference_iterator(video: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    """Keep inference mode enabled while encode_video lazily consumes VAE chunks."""
    with torch.inference_mode():
        yield from video


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def generate(
    pipeline: TI2VidTwoStagesPipeline,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_frames: int,
    stage_1_steps: int,
) -> tuple[Iterator[torch.Tensor], Audio, float]:
    """Run the repository's native full-model Stage 1 + distilled Stage 2 sequence."""
    started = time.perf_counter()
    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
        frame_rate=FRAME_RATE,
        num_inference_steps=stage_1_steps,
        video_guider_params=MultiModalGuiderParams(),
        audio_guider_params=MultiModalGuiderParams(),
        images=[],
        tiling_config=TilingConfig.default(),
    )
    _synchronize()
    return _inference_iterator(video), audio, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--stage-1-steps", type=int, default=DEFAULT_STAGE_1_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=[mode.value for mode in OffloadMode], default=DEFAULT_OFFLOAD.value)
    parser.add_argument(
        "--output-path", type=Path, default=Path("outputs/full-model-two-stage/full-model-two-stage.mp4")
    )
    parser.add_argument(
        "--metrics-path", type=Path, default=Path("outputs/full-model-two-stage/full-model-two-stage.metrics.json")
    )
    args = parser.parse_args()
    if args.stage_1_steps <= 0:
        raise ValueError("stage-1-steps must be positive")

    checkpoint_path = args.model_root / CHECKPOINT_RELATIVE_PATH
    distilled_lora_path = args.model_root / DISTILLED_LORA_RELATIVE_PATH
    upsampler_path = args.model_root / SPATIAL_UPSAMPLER_RELATIVE_PATH
    gemma_root = args.model_root / GEMMA_RELATIVE_PATH
    for path in (checkpoint_path, distilled_lora_path, upsampler_path):
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
    if not gemma_root.is_dir():
        raise FileNotFoundError(f"Gemma directory not found: {gemma_root}")

    num_frames = duration_to_num_frames(args.duration_seconds)
    total_started = time.perf_counter()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=str(checkpoint_path),
        distilled_lora=[LoraPathStrengthAndSDOps(str(distilled_lora_path), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
        spatial_upsampler_path=str(upsampler_path),
        gemma_root=str(gemma_root),
        loras=[],
        offload_mode=OffloadMode(args.offload),
    )
    video, audio, generation_seconds = generate(
        pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_frames=num_frames,
        stage_1_steps=args.stage_1_steps,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_started = time.perf_counter()
    encode_video(
        video=video,
        fps=FRAME_RATE,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
    )
    _synchronize()
    metrics = {
        "checkpoint": str(checkpoint_path),
        "offload": args.offload,
        "stage_1_steps": args.stage_1_steps,
        "stage_2_steps": 3,
        "seed": args.seed,
        "frames": num_frames,
        "fps": FRAME_RATE,
        "output_resolution": [WIDTH, HEIGHT],
        "generation_seconds": round(generation_seconds, 3),
        "encode_seconds": round(time.perf_counter() - encode_started, 3),
        "total_seconds": round(time.perf_counter() - total_started, 3),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
