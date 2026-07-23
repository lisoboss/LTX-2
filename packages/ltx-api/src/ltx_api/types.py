"""Public, LTX-agnostic data structures for the video creation API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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
class FastPreviewRequest:
    prompt: str
    duration_seconds: float
    resolution: VideoResolution
    seed: int | None


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
