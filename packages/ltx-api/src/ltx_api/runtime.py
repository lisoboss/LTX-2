"""Shared model locations and media publication utilities."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

from ltx_api.types import StageMetrics, VideoResolution
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio, SpatioTemporalScaleFactors
from ltx_pipelines.utils.media_io import encode_video

FRAME_RATE = 24.0


@dataclass(frozen=True)
class ModelPaths:
    root: Path

    @property
    def gemma(self) -> Path:
        return self.root / "gemma-3-12b-it-qat-q4_0-unquantized"

    @property
    def distilled(self) -> Path:
        return self.root / "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"

    @property
    def dev(self) -> Path:
        return self.root / "LTX-2.3/ltx-2.3-22b-dev.safetensors"

    @property
    def distilled_lora(self) -> Path:
        return self.root / "LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

    @property
    def upsampler(self) -> Path:
        return self.root / "LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

    @property
    def union_control(self) -> Path:
        return self.root / "LoRA/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"


def num_frames(duration_seconds: float) -> int:
    requested = round(duration_seconds * FRAME_RATE)
    scale = SpatioTemporalScaleFactors.default().time
    frames = ((requested - 1) // scale) * scale + 1
    if frames < scale + 1:
        raise ValueError(f"duration_seconds must produce at least {scale + 1} frames")
    return frames


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def inference_video(video: Iterator[torch.Tensor]) -> Iterator[Iterator[torch.Tensor]]:
    with torch.inference_mode():
        yield video


def encode_atomically(video: Iterator[torch.Tensor], audio: Audio, frames: int, output_path: Path) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.mp4")
    started = time.perf_counter()
    try:
        with inference_video(video) as guarded_video:
            encode_video(
                video=guarded_video,
                fps=FRAME_RATE,
                audio=audio,
                output_path=temporary,
                video_chunks_number=get_video_chunks_number(frames, TilingConfig.default()),
            )
        synchronize()
        temporary.replace(output_path)
        return time.perf_counter() - started
    finally:
        temporary.unlink(missing_ok=True)


def metrics(
    *,
    prompt_seconds: float,
    sample_seconds: float,
    encode_seconds: float,
    total_seconds: float,
    frames: int,
    resolution: VideoResolution,
) -> StageMetrics:
    return StageMetrics(prompt_seconds, sample_seconds, encode_seconds, total_seconds, frames, FRAME_RATE, resolution)
