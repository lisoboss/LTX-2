"""Public, LTX-agnostic data structures for the video creation API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path


class ModelKind(StrEnum):
    DEV = "dev"
    DISTILLED = "distilled"


class QualityPreset(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    HIGH = "high"


@dataclass(frozen=True)
class VideoResolution:
    """Final production resolution for the two-stage workflow."""

    width: int
    height: int

    def __post_init__(self) -> None:
        values = (self.width, self.height)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("width and height must be positive integers (bool is not allowed)")
        if self.width % 64 or self.height % 64:
            suggestion = VideoResolution(_round_up_64(self.width), _round_up_64(self.height))
            raise ValueError(
                f"width and height must be multiples of 64; got {self.width}x{self.height}. "
                f"Minimum compatible recommendation: {suggestion!r}"
            )


def _round_up_64(value: int) -> int:
    return max(64, ((value + 63) // 64) * 64)


@dataclass(frozen=True)
class ImageKeyframe:
    """One user-approved image that constrains a particular output video frame."""

    image_path: Path
    frame_index: int
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.image_path, Path):
            raise TypeError("image_path must be a pathlib.Path")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if (
            not isinstance(self.strength, (int, float))
            or isinstance(self.strength, bool)
            or not isfinite(self.strength)
        ):
            raise ValueError("strength must be a finite number from 0 (exclusive) to 1 (inclusive)")
        if not 0 < self.strength <= 1:
            raise ValueError("strength must be from 0 (exclusive) to 1 (inclusive)")


@dataclass(frozen=True)
class FastPreviewRequest:
    prompt: str
    duration_seconds: float
    resolution: VideoResolution
    seed: int | None
    keyframes: tuple[ImageKeyframe, ...] = ()


@dataclass(frozen=True)
class QualityPreviewRequest:
    prompt: str
    negative_prompt: str
    duration_seconds: float
    resolution: VideoResolution
    seed: int | None
    quality: QualityPreset
    num_inference_steps: int
    video_cfg_scale: float
    audio_cfg_scale: float
    keyframes: tuple[ImageKeyframe, ...] = ()


@dataclass(frozen=True)
class StageMetrics:
    prompt_seconds: float
    sample_seconds: float
    encode_seconds: float
    total_seconds: float
    frames: int
    frame_rate: float
    output_resolution: VideoResolution


@dataclass(frozen=True)
class PreviewResult:
    preview_path: Path
    artifact_path: Path
    model: ModelKind
    effective_request: FastPreviewRequest | QualityPreviewRequest
    metrics: StageMetrics


@dataclass(frozen=True)
class ProductionResult:
    production_path: Path
    artifact_path: Path
    model: ModelKind
    metrics: StageMetrics


@dataclass(frozen=True)
class EnhanceResult:
    enhanced_path: Path
    production_path: Path
    ic_lora_path: Path
    metrics: StageMetrics
