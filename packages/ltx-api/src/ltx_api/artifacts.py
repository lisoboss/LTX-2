"""Portable, versioned stage-one artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

from ltx_api.types import ModelKind, VideoResolution

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class StageOneArtifact:
    model: ModelKind
    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    generator_state: torch.Tensor
    prompt: str
    negative_prompt: str
    seed: int
    resolution: VideoResolution
    num_frames: int
    frame_rate: float
    num_inference_steps: int | None = None
    video_cfg_scale: float | None = None
    audio_cfg_scale: float | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload = {
            **self.__dict__,
            "version": ARTIFACT_VERSION,
            "model": self.model.value,
            "resolution": (self.resolution.width, self.resolution.height),
        }
        payload.update(
            {
                key: value.detach().cpu().contiguous()
                for key, value in payload.items()
                if isinstance(value, torch.Tensor)
            }
        )
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> StageOneArtifact:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.pop("version", None) != ARTIFACT_VERSION:
            raise ValueError("unsupported stage-one artifact")
        width, height = payload.pop("resolution")
        payload["resolution"] = VideoResolution(width, height)
        payload["model"] = ModelKind(payload["model"])
        return cls(**payload)
