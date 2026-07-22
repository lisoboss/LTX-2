"""Preview-to-production adapters built from the existing LTX-2 pipelines.

This module deliberately composes the public pipeline building blocks instead of
changing the existing CLI-oriented pipeline implementations.  Runtime callers
can persist a stage-1 artifact and later refine it without repeating sampling.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio, SpatioTemporalScaleFactors
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

ARTIFACT_VERSION = 1
PREVIEW_HEIGHT = 384
PREVIEW_WIDTH = 640
PRODUCTION_HEIGHT = PREVIEW_HEIGHT * 2
PRODUCTION_WIDTH = PREVIEW_WIDTH * 2
FRAME_RATE = 24.0
DEFAULT_SEED = 10
DEFAULT_OFFLOAD_MODE = OffloadMode.CPU

DISTILLED_CHECKPOINT_RELATIVE_PATH = Path("LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors")
SPATIAL_UPSAMPLER_RELATIVE_PATH = Path("LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
DISTILLED_LORA_RELATIVE_PATH = Path("LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
GEMMA_RELATIVE_PATH = Path("gemma-3-12b-it-qat-q4_0-unquantized")
IC_LORA_RELATIVE_PATH = Path("LTX-2.3/ltx-2.3-22b-ic-lora-union-control.safetensors")


def duration_to_num_frames(duration_seconds: float, frame_rate: float = FRAME_RATE) -> int:
    """Convert seconds to the largest LTX-compatible frame count not exceeding it."""
    if not torch.isfinite(torch.tensor(duration_seconds)) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be a finite positive number")
    if not torch.isfinite(torch.tensor(frame_rate)) or frame_rate <= 0:
        raise ValueError("frame_rate must be a finite positive number")

    requested_frames = max(1, round(duration_seconds * frame_rate))
    time_scale = SpatioTemporalScaleFactors.default().time
    frames = ((requested_frames - 1) // time_scale) * time_scale + 1
    if frames < time_scale + 1:
        raise ValueError(f"duration_seconds is too short; it must produce at least {time_scale + 1} frames")
    return frames


@dataclass(frozen=True)
class DistilledStage1Artifact:
    """Portable CPU latent state emitted by distilled stage 1."""

    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    prompt: str
    seed: int
    height: int
    width: int
    num_frames: int
    frame_rate: float
    version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        _validate_artifact(self)

    def save(self, path: Path) -> None:
        """Persist a device-independent artifact. Tensors are always copied to CPU."""
        path = Path(path)
        payload = {
            "version": self.version,
            "video_latent": self.video_latent.detach().to(device="cpu").contiguous(),
            "audio_latent": self.audio_latent.detach().to(device="cpu").contiguous(),
            "prompt": self.prompt,
            "seed": self.seed,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "frame_rate": self.frame_rate,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> DistilledStage1Artifact:
        """Load a CPU artifact without allowing arbitrary pickle execution."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("stage-1 artifact payload must be a dictionary")
        required = {
            "version",
            "video_latent",
            "audio_latent",
            "prompt",
            "seed",
            "height",
            "width",
            "num_frames",
            "frame_rate",
        }
        if payload.keys() != required:
            raise ValueError("stage-1 artifact has an unsupported schema")
        return cls(**payload)


@dataclass(frozen=True)
class FastVideoResult:
    """Decoded preview media plus its reusable latent artifact."""

    video: Iterator[torch.Tensor]
    audio: Audio
    artifact: DistilledStage1Artifact


class PreviewProductionPipeline:
    """Expose stage boundaries by composing an existing :class:`DistilledPipeline`."""

    def __init__(self, pipeline: DistilledPipeline) -> None:
        self._pipeline = pipeline

    @torch.inference_mode()
    def run_stage_1(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput] | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
    ) -> FastVideoResult:
        """Sample and decode only distilled stage 1 at half target resolution."""
        assert_resolution(height=height, width=width, is_two_stage=True)
        _validate_frame_metadata(num_frames, frame_rate)
        images = images or []
        pipeline = self._pipeline
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        (context,) = pipeline.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if images else None,
        )
        stage_height, stage_width = height // 2, width // 2
        conditionings = pipeline.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=stage_height,
                width=stage_width,
                video_encoder=encoder,
                dtype=pipeline.dtype,
                device=pipeline.device,
            )
        )
        video_state, audio_state = pipeline.stage(
            denoiser=SimpleDenoiser(context.video_encoding, context.audio_encoding),
            sigmas=stage_1_sigmas.to(dtype=torch.float32, device=pipeline.device),
            noiser=noiser,
            width=stage_width,
            height=stage_height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=context.video_encoding, conditionings=conditionings),
            audio=ModalitySpec(context=context.audio_encoding),
        )
        artifact = DistilledStage1Artifact(
            video_latent=video_state.latent.detach().to(device="cpu"),
            audio_latent=audio_state.latent.detach().to(device="cpu"),
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )
        return FastVideoResult(
            video=pipeline.video_decoder(video_state.latent, tiling_config, generator),
            audio=pipeline.audio_decoder(audio_state.latent),
            artifact=artifact,
        )

    @torch.inference_mode()
    def run_stage_2(
        self,
        artifact: DistilledStage1Artifact,
        *,
        images: list[ImageConditioningInput] | None = None,
        tiling_config: TilingConfig | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Upsample and refine an existing stage-1 artifact without sampling stage 1."""
        _validate_artifact(artifact)
        pipeline = self._pipeline
        images = images or []
        generator = torch.Generator(device=pipeline.device).manual_seed(artifact.seed)
        noiser = GaussianNoiser(generator=generator)
        (context,) = pipeline.prompt_encoder(
            [artifact.prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=images[0][0] if images else None,
        )
        video_latent = artifact.video_latent.to(device=pipeline.device, dtype=pipeline.dtype)
        audio_latent = artifact.audio_latent.to(device=pipeline.device, dtype=pipeline.dtype)
        upscaled_video_latent = pipeline.upsampler(video_latent[:1])
        sigmas = stage_2_sigmas.to(dtype=torch.float32, device=pipeline.device)
        conditionings = pipeline.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=artifact.height,
                width=artifact.width,
                video_encoder=encoder,
                dtype=pipeline.dtype,
                device=pipeline.device,
            )
        )
        video_state, audio_state = pipeline.stage(
            denoiser=SimpleDenoiser(context.video_encoding, context.audio_encoding),
            sigmas=sigmas,
            noiser=noiser,
            width=artifact.width,
            height=artifact.height,
            frames=artifact.num_frames,
            fps=artifact.frame_rate,
            video=ModalitySpec(
                context=context.video_encoding,
                conditionings=conditionings,
                noise_scale=sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=context.audio_encoding,
                noise_scale=sigmas[0].item(),
                initial_latent=audio_latent,
            ),
        )
        return (
            pipeline.video_decoder(video_state.latent, tiling_config, generator),
            pipeline.audio_decoder(audio_state.latent),
        )


class PreviewVideoModifyPipeline:
    """Production-mode preview editing adapter for an existing IC-LoRA pipeline."""

    def __init__(self, pipeline: ICLoraPipeline) -> None:
        self._pipeline = pipeline

    @torch.inference_mode()
    def modify_preview(
        self,
        *,
        preview_video_path: str | Path,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Modify a preview while always retaining the IC-LoRA production stage."""
        return self._pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            video_conditioning=[(str(preview_video_path), 1.0)],
            tiling_config=tiling_config,
            skip_stage_2=False,
        )


class LTX2FastVideo:
    """Model-root-only low-resolution preview feature."""

    def __init__(self, model_root: Path, offload_mode: OffloadMode = DEFAULT_OFFLOAD_MODE) -> None:
        self.model_root = Path(model_root)
        self._adapter = PreviewProductionPipeline(_make_distilled_pipeline(self.model_root, offload_mode))

    def generate(self, *, prompt: str, duration_seconds: float, seed: int | None = None) -> FastVideoResult:
        return self._adapter.run_stage_1(
            prompt=prompt,
            seed=DEFAULT_SEED if seed is None else seed,
            height=PRODUCTION_HEIGHT,
            width=PRODUCTION_WIDTH,
            num_frames=duration_to_num_frames(duration_seconds),
            frame_rate=FRAME_RATE,
        )


class LTX2HighResolutionVideo:
    """Model-root-only stage-2 production feature."""

    def __init__(self, model_root: Path, offload_mode: OffloadMode = DEFAULT_OFFLOAD_MODE) -> None:
        self.model_root = Path(model_root)
        self._adapter = PreviewProductionPipeline(_make_distilled_pipeline(self.model_root, offload_mode))

    def generate(self, *, artifact: DistilledStage1Artifact) -> tuple[Iterator[torch.Tensor], Audio]:
        if (artifact.height, artifact.width, artifact.frame_rate) != (PRODUCTION_HEIGHT, PRODUCTION_WIDTH, FRAME_RATE):
            raise ValueError("artifact does not match the fixed production resolution or frame rate")
        return self._adapter.run_stage_2(artifact)


class LTX2VideoModify:
    """Model-root-only IC-LoRA preview modification feature."""

    def __init__(self, model_root: Path, offload_mode: OffloadMode = DEFAULT_OFFLOAD_MODE) -> None:
        self.model_root = Path(model_root)
        self._adapter = PreviewVideoModifyPipeline(_make_ic_lora_pipeline(self.model_root, offload_mode))

    def generate(
        self,
        *,
        preview_video_path: str | Path,
        prompt: str,
        duration_seconds: float,
        seed: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        return self._adapter.modify_preview(
            preview_video_path=preview_video_path,
            prompt=prompt,
            seed=DEFAULT_SEED if seed is None else seed,
            height=PRODUCTION_HEIGHT,
            width=PRODUCTION_WIDTH,
            num_frames=duration_to_num_frames(duration_seconds),
            frame_rate=FRAME_RATE,
        )


def _make_distilled_pipeline(model_root: Path, offload_mode: OffloadMode) -> DistilledPipeline:
    return DistilledPipeline(
        distilled_checkpoint_path=str(model_root / DISTILLED_CHECKPOINT_RELATIVE_PATH),
        gemma_root=str(model_root / GEMMA_RELATIVE_PATH),
        spatial_upsampler_path=str(model_root / SPATIAL_UPSAMPLER_RELATIVE_PATH),
        loras=[
            LoraPathStrengthAndSDOps(
                str(model_root / DISTILLED_LORA_RELATIVE_PATH),
                1.0,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        offload_mode=offload_mode,
    )


def _make_ic_lora_pipeline(model_root: Path, offload_mode: OffloadMode) -> ICLoraPipeline:
    lora_path = model_root / IC_LORA_RELATIVE_PATH
    return ICLoraPipeline(
        distilled_checkpoint_path=str(model_root / DISTILLED_CHECKPOINT_RELATIVE_PATH),
        gemma_root=str(model_root / GEMMA_RELATIVE_PATH),
        spatial_upsampler_path=str(model_root / SPATIAL_UPSAMPLER_RELATIVE_PATH),
        loras=[LoraPathStrengthAndSDOps(str(lora_path), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
        offload_mode=offload_mode,
    )


def _validate_artifact(artifact: DistilledStage1Artifact) -> None:
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(f"unsupported stage-1 artifact version: {artifact.version}")
    if not isinstance(artifact.video_latent, torch.Tensor) or artifact.video_latent.ndim != 5:
        raise ValueError("artifact video_latent must be a five-dimensional tensor")
    if not isinstance(artifact.audio_latent, torch.Tensor) or artifact.audio_latent.ndim < 3:
        raise ValueError("artifact audio_latent must be a tensor with at least three dimensions")
    if artifact.video_latent.shape[0] != 1 or artifact.audio_latent.shape[0] != 1:
        raise ValueError("stage-1 artifacts must have batch size 1")
    assert_resolution(height=artifact.height, width=artifact.width, is_two_stage=True)
    _validate_frame_metadata(artifact.num_frames, artifact.frame_rate)


def _validate_frame_metadata(num_frames: int, frame_rate: float) -> None:
    time_scale = SpatioTemporalScaleFactors.default().time
    if num_frames < time_scale + 1 or (num_frames - 1) % time_scale != 0:
        raise ValueError(f"num_frames must satisfy {time_scale}k + 1")
    if not torch.isfinite(torch.tensor(frame_rate)) or frame_rate <= 0:
        raise ValueError("frame_rate must be a finite positive number")


def main() -> None:
    """Run one preview/production flow from the repository root for local validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fast_parser = subparsers.add_parser("fast", help="Generate a stage-1 preview and artifact")
    _add_model_arguments(fast_parser)
    fast_parser.add_argument("--prompt", required=True)
    fast_parser.add_argument("--duration-seconds", required=True, type=float)
    fast_parser.add_argument("--artifact-path", required=True, type=Path)
    fast_parser.add_argument("--output-path", required=True, type=Path)
    fast_parser.add_argument("--seed", type=int)

    production_parser = subparsers.add_parser("production", help="Refine a saved stage-1 artifact")
    _add_model_arguments(production_parser)
    production_parser.add_argument("--artifact-path", required=True, type=Path)
    production_parser.add_argument("--output-path", required=True, type=Path)

    modify_parser = subparsers.add_parser("modify", help="Modify a preview through IC-LoRA")
    _add_model_arguments(modify_parser)
    modify_parser.add_argument("--preview-video-path", required=True, type=Path)
    modify_parser.add_argument("--prompt", required=True)
    modify_parser.add_argument("--duration-seconds", required=True, type=float)
    modify_parser.add_argument("--output-path", required=True, type=Path)
    modify_parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    if args.command == "fast":
        result = LTX2FastVideo(args.model_root, args.offload).generate(
            prompt=args.prompt,
            duration_seconds=args.duration_seconds,
            seed=args.seed,
        )
        result.artifact.save(args.artifact_path)
        _encode(result.video, result.audio, result.artifact.num_frames, result.artifact.frame_rate, args.output_path)
    elif args.command == "production":
        artifact = DistilledStage1Artifact.load(args.artifact_path)
        video, audio = LTX2HighResolutionVideo(args.model_root, args.offload).generate(artifact=artifact)
        _encode(video, audio, artifact.num_frames, artifact.frame_rate, args.output_path)
    else:
        video, audio = LTX2VideoModify(args.model_root, args.offload).generate(
            preview_video_path=args.preview_video_path,
            prompt=args.prompt,
            duration_seconds=args.duration_seconds,
            seed=args.seed,
        )
        _encode(video, audio, duration_to_num_frames(args.duration_seconds), FRAME_RATE, args.output_path)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument(
        "--offload",
        type=OffloadMode,
        choices=list(OffloadMode),
        default=DEFAULT_OFFLOAD_MODE,
        help="Weight streaming mode: cpu (default, about 5 GB VRAM), disk, or none (about 28 GB VRAM).",
    )


def _encode(
    video: Iterator[torch.Tensor],
    audio: Audio,
    num_frames: int,
    frame_rate: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
    )


if __name__ == "__main__":
    main()
