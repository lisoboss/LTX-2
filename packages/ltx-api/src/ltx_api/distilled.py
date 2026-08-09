"""Distilled fast-preview and continuation adapters."""

from __future__ import annotations

import time
from collections.abc import Iterator

import torch

from ltx_api.artifacts import StageOneArtifact
from ltx_api.runtime import FRAME_RATE, ModelPaths, num_frames, synchronize
from ltx_api.types import FastPreviewRequest, ImageKeyframe, ModelKind
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import Audio
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class DistilledAdapter:
    def __init__(self, paths: ModelPaths, offload: OffloadMode) -> None:
        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=str(paths.distilled),
            gemma_root=str(paths.gemma),
            spatial_upsampler_path=str(paths.upsampler),
            loras=[LoraPathStrengthAndSDOps(str(paths.distilled_lora), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
            offload_mode=offload,
        )

    @staticmethod
    def _image_inputs(keyframes: tuple[ImageKeyframe, ...]) -> list[ImageConditioningInput]:
        return [
            ImageConditioningInput(str(item.image_path), item.frame_index, item.strength) for item in keyframes
        ]

    @torch.inference_mode()
    def preview(
        self, request: FastPreviewRequest
    ) -> tuple[Iterator[torch.Tensor], Audio, StageOneArtifact, float, float]:
        pipeline = self._pipeline
        resolution = request.resolution
        assert_resolution(height=resolution.height, width=resolution.width, is_two_stage=True)
        frames = num_frames(request.duration_seconds)
        seed = 42 if request.seed is None else request.seed
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        prompt_started = time.perf_counter()
        (context,) = pipeline.prompt_encoder([request.prompt], enhance_first_prompt=False, enhance_prompt_image=None)
        synchronize()
        prompt_seconds = time.perf_counter() - prompt_started
        sample_started = time.perf_counter()
        images = self._image_inputs(request.keyframes)
        conditionings = pipeline.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=resolution.height // 2,
                width=resolution.width // 2,
                video_encoder=encoder,
                dtype=pipeline.dtype,
                device=pipeline.device,
            )
        )
        video_state, audio_state = pipeline.stage(
            denoiser=SimpleDenoiser(context.video_encoding, context.audio_encoding),
            sigmas=DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device),
            noiser=noiser,
            width=resolution.width // 2,
            height=resolution.height // 2,
            frames=frames,
            fps=FRAME_RATE,
            video=ModalitySpec(context=context.video_encoding, conditionings=conditionings),
            audio=ModalitySpec(context=context.audio_encoding),
        )
        synchronize()
        artifact = StageOneArtifact(
            ModelKind.DISTILLED,
            video_state.latent.detach().cpu(),
            audio_state.latent.detach().cpu(),
            generator.get_state().detach().cpu(),
            request.prompt,
            "",
            seed,
            resolution,
            frames,
            FRAME_RATE,
            keyframes=request.keyframes,
        )
        return (
            pipeline.video_decoder(video_state.latent, None, generator),
            pipeline.audio_decoder(audio_state.latent),
            artifact,
            prompt_seconds,
            time.perf_counter() - sample_started,
        )

    @torch.inference_mode()
    def production(self, artifact: StageOneArtifact) -> tuple[Iterator[torch.Tensor], Audio, float, float]:
        if artifact.model is not ModelKind.DISTILLED:
            raise ValueError("a distilled production requires a fast-preview artifact")
        started = time.perf_counter()
        pipeline = self._pipeline
        generator = torch.Generator(device=pipeline.device).manual_seed(artifact.seed)
        generator.set_state(artifact.generator_state)
        noiser = GaussianNoiser(generator=generator)
        prompt_started = time.perf_counter()
        (context,) = pipeline.prompt_encoder([artifact.prompt], enhance_first_prompt=False, enhance_prompt_image=None)
        synchronize()
        upscaled = pipeline.upsampler(artifact.video_latent.to(pipeline.device, pipeline.dtype)[:1])
        images = self._image_inputs(artifact.keyframes)
        conditionings = pipeline.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=artifact.resolution.height,
                width=artifact.resolution.width,
                video_encoder=encoder,
                dtype=pipeline.dtype,
                device=pipeline.device,
            )
        )
        video_state, audio_state = pipeline.stage(
            denoiser=SimpleDenoiser(context.video_encoding, context.audio_encoding),
            sigmas=STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device),
            noiser=noiser,
            width=artifact.resolution.width,
            height=artifact.resolution.height,
            frames=artifact.num_frames,
            fps=artifact.frame_rate,
            video=ModalitySpec(
                context=context.video_encoding,
                conditionings=conditionings,
                noise_scale=STAGE_2_DISTILLED_SIGMAS[0].item(),
                initial_latent=upscaled,
            ),
            audio=ModalitySpec(
                context=context.audio_encoding,
                noise_scale=STAGE_2_DISTILLED_SIGMAS[0].item(),
                initial_latent=artifact.audio_latent.to(pipeline.device, pipeline.dtype),
            ),
        )
        synchronize()
        return (
            pipeline.video_decoder(video_state.latent, None, generator),
            pipeline.audio_decoder(audio_state.latent),
            time.perf_counter() - prompt_started,
            time.perf_counter() - started,
        )
