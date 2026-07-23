from __future__ import annotations

import pytest
from ltx_api.types import VideoResolution


def test_resolution_suggests_smallest_compatible_value() -> None:
    with pytest.raises(ValueError, match=r"VideoResolution\(width=1024, height=768\)"):
        VideoResolution(width=1000, height=720)


def test_resolution_rejects_bool() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        VideoResolution(width=True, height=768)
