"""Friendly builders which prevent incompatible model parameters by construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from ltx_api.presets import QUALITY_PRESETS
from ltx_api.types import FastPreviewRequest, QualityPreset, QualityPreviewRequest, VideoResolution

_DEFAULT_RESOLUTION = VideoResolution(1280, 768)


@dataclass
class _BasePreviewBuilder:
    _prompt: str | None = None
    _duration_seconds: float = 5.0
    _resolution: VideoResolution = _DEFAULT_RESOLUTION
    _seed: int | None = 42

    def prompt(self, value: str) -> Self:
        self._prompt = value
        return self

    def duration_seconds(self, value: float) -> Self:
        self._duration_seconds = value
        return self

    def resolution(self, width: int, height: int) -> Self:
        self._resolution = VideoResolution(width, height)
        return self

    def seed(self, value: int | None) -> Self:
        self._seed = value
        return self

    def _validate(self) -> str:
        if not self._prompt or not self._prompt.strip():
            raise ValueError("prompt must be set before build()")
        if self._duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        return self._prompt


class FastPreviewBuilder(_BasePreviewBuilder):
    def build(self) -> FastPreviewRequest:
        return FastPreviewRequest(self._validate(), self._duration_seconds, self._resolution, self._seed)


@dataclass
class QualityPreviewBuilder(_BasePreviewBuilder):
    _negative_prompt: str = ""
    _quality: QualityPreset = QualityPreset.STANDARD
    _num_inference_steps: int | None = None
    _video_cfg_scale: float | None = None
    _audio_cfg_scale: float | None = None

    def negative_prompt(self, value: str) -> Self:
        self._negative_prompt = value
        return self

    def quality(self, value: QualityPreset | str) -> Self:
        self._quality = QualityPreset(value)
        return self

    def num_inference_steps(self, value: int) -> Self:
        self._num_inference_steps = value
        return self

    def video_cfg_scale(self, value: float) -> Self:
        self._video_cfg_scale = value
        return self

    def audio_cfg_scale(self, value: float) -> Self:
        self._audio_cfg_scale = value
        return self

    def build(self) -> QualityPreviewRequest:
        prompt = self._validate()
        preset = QUALITY_PRESETS[self._quality]
        steps = self._num_inference_steps if self._num_inference_steps is not None else preset.num_inference_steps
        video_cfg = self._video_cfg_scale if self._video_cfg_scale is not None else preset.video_cfg_scale
        audio_cfg = self._audio_cfg_scale if self._audio_cfg_scale is not None else preset.audio_cfg_scale
        if steps <= 0 or video_cfg <= 0 or audio_cfg <= 0:
            raise ValueError("steps and CFG scales must be positive")
        return QualityPreviewRequest(
            prompt,
            self._negative_prompt,
            self._duration_seconds,
            self._resolution,
            self._seed,
            self._quality,
            steps,
            video_cfg,
            audio_cfg,
        )
