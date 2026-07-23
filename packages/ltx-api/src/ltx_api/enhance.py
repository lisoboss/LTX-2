"""Production-video enhancement through the fixed Union Control IC-LoRA."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import torch

from ltx_api.runtime import FRAME_RATE, ModelPaths, num_frames, synchronize
from ltx_api.types import VideoResolution
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import Audio
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.types import OffloadMode


class EnhanceAdapter:
    def __init__(self, paths: ModelPaths, offload: OffloadMode) -> None:
        self.paths = paths
        self.pipeline = ICLoraPipeline(
            distilled_checkpoint_path=str(paths.distilled),
            gemma_root=str(paths.gemma),
            spatial_upsampler_path=str(paths.upsampler),
            loras=[LoraPathStrengthAndSDOps(str(paths.union_control), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
            offload_mode=offload,
        )

    @torch.inference_mode()
    def run(
        self,
        production_path: Path,
        prompt: str,
        duration_seconds: float,
        seed: int | None,
        resolution: VideoResolution,
    ) -> tuple[Iterator[torch.Tensor], Audio, int, float]:
        if not production_path.is_file():
            raise FileNotFoundError(f"production video not found: {production_path}")
        if not self.paths.union_control.is_file():
            raise FileNotFoundError(f"Union Control IC-LoRA not found: {self.paths.union_control}")
        frames = num_frames(duration_seconds)
        started = time.perf_counter()
        video, audio = self.pipeline(
            prompt=prompt,
            seed=42 if seed is None else seed,
            height=resolution.height,
            width=resolution.width,
            num_frames=frames,
            frame_rate=FRAME_RATE,
            images=[],
            video_conditioning=[(str(production_path), 1.0)],
            skip_stage_2=False,
        )
        synchronize()
        return video, audio, frames, time.perf_counter() - started
