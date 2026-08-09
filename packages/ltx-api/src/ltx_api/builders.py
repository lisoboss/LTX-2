"""Friendly builders which prevent incompatible model parameters by construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Self

from ltx_api.presets import QUALITY_PRESETS
from ltx_api.runtime import num_frames
from ltx_api.types import FastPreviewRequest, ImageKeyframe, QualityPreset, QualityPreviewRequest, VideoResolution

_DEFAULT_RESOLUTION = VideoResolution(1280, 768)


@dataclass
class _BasePreviewBuilder:
    _prompt: str | None = None
    _duration_seconds: float = 5.0
    _resolution: VideoResolution = _DEFAULT_RESOLUTION
    _seed: int | None = 42
    _first_frame: tuple[Path, float] | None = None
    _middle_frames: list[tuple[Path, float, float]] | None = None
    _last_frame: tuple[Path, float] | None = None

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

    def first_frame(self, image_path: Path, *, strength: float = 1.0) -> Self:
        """Use an approved image as the exact opening frame of the video."""
        self._first_frame = (Path(image_path), strength)
        return self

    def middle_frame(self, image_path: Path, *, at_seconds: float, strength: float = 1.0) -> Self:
        """Guide a frame inside the video at ``at_seconds`` from its start."""
        if self._middle_frames is None:
            self._middle_frames = []
        self._middle_frames.append((Path(image_path), at_seconds, strength))
        return self

    def last_frame(self, image_path: Path, *, strength: float = 1.0) -> Self:
        """Guide the final frame of the generated video."""
        self._last_frame = (Path(image_path), strength)
        return self

    def _validate(self) -> str:
        if not self._prompt or not self._prompt.strip():
            raise ValueError("prompt must be set before build()")
        if self._duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        return self._prompt

    def _keyframes(self) -> tuple[ImageKeyframe, ...]:
        frames = num_frames(self._duration_seconds)
        result: list[ImageKeyframe] = []
        if self._first_frame is not None:
            path, strength = self._first_frame
            result.append(ImageKeyframe(path, 0, strength))
        for path, at_seconds, strength in self._middle_frames or []:
            if not isinstance(at_seconds, (int, float)) or isinstance(at_seconds, bool) or not isfinite(at_seconds):
                raise ValueError("middle_frame at_seconds must be a finite number")
            frame_index = round(at_seconds * 24)
            if frame_index <= 0 or frame_index >= frames - 1:
                max_seconds = (frames - 1) / 24
                raise ValueError(f"middle_frame at_seconds must be between 0 and {max_seconds:g} (exclusive)")
            result.append(ImageKeyframe(path, frame_index, strength))
        if self._last_frame is not None:
            path, strength = self._last_frame
            result.append(ImageKeyframe(path, frames - 1, strength))

        indexes = [keyframe.frame_index for keyframe in result]
        if len(indexes) != len(set(indexes)):
            raise ValueError("keyframe images must target different video frames")
        missing = [keyframe.image_path for keyframe in result if not keyframe.image_path.is_file()]
        if missing:
            raise FileNotFoundError(f"keyframe image does not exist: {missing[0]}")
        return tuple(result)


class FastPreviewBuilder(_BasePreviewBuilder):
    def build(self) -> FastPreviewRequest:
        return FastPreviewRequest(
            self._validate(), self._duration_seconds, self._resolution, self._seed, self._keyframes()
        )


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
            self._keyframes(),
        )
