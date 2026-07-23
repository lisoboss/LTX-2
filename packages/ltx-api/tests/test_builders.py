from __future__ import annotations

import pytest
from ltx_api import FastPreviewBuilder, QualityPreviewBuilder


def test_fast_builder_defaults() -> None:
    request = FastPreviewBuilder().prompt("fox").build()
    assert (request.duration_seconds, request.resolution.width, request.resolution.height, request.seed) == (
        5.0,
        1280,
        768,
        42,
    )


def test_quality_override_wins_over_preset() -> None:
    request = QualityPreviewBuilder().prompt("fox").quality("fast").num_inference_steps(24).build()
    assert (request.num_inference_steps, request.video_cfg_scale, request.audio_cfg_scale) == (24, 2.5, 7.0)


def test_prompt_is_required() -> None:
    with pytest.raises(ValueError, match="prompt"):
        FastPreviewBuilder().build()
