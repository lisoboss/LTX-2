"""The friendly three-stage public facade."""

from __future__ import annotations

import time
from pathlib import Path

from ltx_api.artifacts import StageOneArtifact
from ltx_api.dev import DevAdapter
from ltx_api.distilled import DistilledAdapter
from ltx_api.enhance import EnhanceAdapter
from ltx_api.runtime import ModelPaths, encode_atomically, metrics
from ltx_api.types import (
    EnhanceResult,
    FastPreviewRequest,
    ModelKind,
    PreviewResult,
    ProductionResult,
    QualityPreviewRequest,
    VideoResolution,
)
from ltx_pipelines.utils.types import OffloadMode

DEFAULT_RESOLUTION = VideoResolution(1280, 768)


class VideoCreator:
    """Create fast previews, production videos, and IC-LoRA enhancements."""

    def __init__(self, model_root: Path, *, offload_mode: OffloadMode = OffloadMode.DISK) -> None:
        self.paths = ModelPaths(Path(model_root))
        self.offload_mode = offload_mode

    def preview(
        self,
        request: FastPreviewRequest | QualityPreviewRequest,
        *,
        artifact_path: Path,
        output_path: Path,
    ) -> PreviewResult:
        adapter = (
            DistilledAdapter(self.paths, self.offload_mode)
            if isinstance(request, FastPreviewRequest)
            else DevAdapter(self.paths, self.offload_mode)
        )
        started = time.perf_counter()
        video, audio, artifact, prompt_seconds, sample_seconds = adapter.preview(request)
        artifact.save(artifact_path)
        encode_seconds = encode_atomically(video, audio, artifact.num_frames, output_path)
        return PreviewResult(
            output_path,
            artifact_path,
            ModelKind.DISTILLED,
            request,
            metrics(
                prompt_seconds=prompt_seconds,
                sample_seconds=sample_seconds,
                encode_seconds=encode_seconds,
                total_seconds=time.perf_counter() - started,
                frames=artifact.num_frames,
                resolution=request.resolution,
            ),
        )

    def create_production(self, *, artifact_path: Path, output_path: Path) -> ProductionResult:
        started = time.perf_counter()
        artifact = StageOneArtifact.load(artifact_path)
        adapter = (
            DistilledAdapter(self.paths, self.offload_mode)
            if artifact.model is ModelKind.DISTILLED
            else DevAdapter(self.paths, self.offload_mode)
        )
        video, audio, prompt_seconds, sample_seconds = adapter.production(artifact)
        encode_seconds = encode_atomically(video, audio, artifact.num_frames, output_path)
        return ProductionResult(
            output_path,
            artifact_path,
            artifact.model,
            metrics(
                prompt_seconds=prompt_seconds,
                sample_seconds=sample_seconds,
                encode_seconds=encode_seconds,
                total_seconds=time.perf_counter() - started,
                frames=artifact.num_frames,
                resolution=artifact.resolution,
            ),
        )

    def enhance(
        self,
        *,
        production_path: Path,
        output_path: Path,
        prompt: str,
        duration_seconds: float,
        resolution: VideoResolution = DEFAULT_RESOLUTION,
        seed: int | None = None,
    ) -> EnhanceResult:
        video, audio, frames, sample_seconds = EnhanceAdapter(self.paths, self.offload_mode).run(
            production_path, prompt, duration_seconds, seed, resolution
        )
        encode_seconds = encode_atomically(video, audio, frames, output_path)
        return EnhanceResult(
            output_path,
            production_path,
            self.paths.union_control,
            metrics(
                prompt_seconds=0.0,
                sample_seconds=sample_seconds,
                encode_seconds=encode_seconds,
                total_seconds=sample_seconds + encode_seconds,
                frames=frames,
                resolution=resolution,
            ),
        )
