"""Safe public quality presets and fixed LTX-2.3 internal guidance values."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_api.types import QualityPreset


@dataclass(frozen=True)
class QualitySettings:
    num_inference_steps: int
    video_cfg_scale: float
    audio_cfg_scale: float


QUALITY_PRESETS = {
    QualityPreset.FAST: QualitySettings(12, 2.5, 7.0),
    QualityPreset.STANDARD: QualitySettings(30, 3.0, 7.0),
    QualityPreset.HIGH: QualitySettings(40, 3.0, 7.0),
}
VIDEO_STG_SCALE = 1.0
AUDIO_STG_SCALE = 1.0
VIDEO_RESCALE_SCALE = 0.7
AUDIO_RESCALE_SCALE = 0.7
MODALITY_SCALE = 3.0
STG_BLOCKS = [28]
