"""Full 22B dev stage-one and continuation adapter."""

from __future__ import annotations

import time
from collections.abc import Iterator

import torch

from ltx_api.artifacts import StageOneArtifact
from ltx_api.presets import (
    AUDIO_RESCALE_SCALE,
    AUDIO_STG_SCALE,
    MODALITY_SCALE,
    STG_BLOCKS,
    VIDEO_RESCALE_SCALE,
    VIDEO_STG_SCALE,
)
from ltx_api.runtime import FRAME_RATE, ModelPaths, num_frames, synchronize
from ltx_api.types import ImageKeyframe, ModelKind, QualityPreviewRequest
from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import combined_image_conditionings
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class DevAdapter:
    def __init__(self, paths: ModelPaths, offload: OffloadMode) -> None:
        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=str(paths.dev),
            distilled_lora=[LoraPathStrengthAndSDOps(str(paths.distilled_lora), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
            spatial_upsampler_path=str(paths.upsampler),
            gemma_root=str(paths.gemma),
            loras=[],
            offload_mode=offload,
        )
        self.scheduler = LTX2Scheduler()

    @staticmethod
    def _image_inputs(keyframes: tuple[ImageKeyframe, ...]) -> list[ImageConditioningInput]:
        return [
            ImageConditioningInput(str(item.image_path), item.frame_index, item.strength) for item in keyframes
        ]

    @torch.inference_mode()
    def preview(
        self, request: QualityPreviewRequest
    ) -> tuple[Iterator[torch.Tensor], Audio, StageOneArtifact, float, float]:
        p = self.pipeline
        seed = 42 if request.seed is None else request.seed
        frames = num_frames(request.duration_seconds)
        generator = torch.Generator(device=p.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        prompt_started = time.perf_counter()
        positive, negative = p.prompt_encoder(
            [request.prompt, request.negative_prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        synchronize()
        shape = VideoPixelShape(
            batch=1,
            frames=frames,
            width=request.resolution.width // 2,
            height=request.resolution.height // 2,
            fps=FRAME_RATE,
        )
        images = self._image_inputs(request.keyframes)
        conditionings = p.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=shape.height,
                width=shape.width,
                video_encoder=encoder,
                dtype=p.dtype,
                device=p.device,
            )
        )
        sample_started = time.perf_counter()
        video_state, audio_state = p.stage_1(
            denoiser=FactoryGuidedDenoiser(
                v_context=positive.video_encoding,
                a_context=positive.audio_encoding,
                video_guider_factory=create_multimodal_guider_factory(
                    MultiModalGuiderParams(
                        cfg_scale=request.video_cfg_scale,
                        stg_scale=VIDEO_STG_SCALE,
                        stg_blocks=STG_BLOCKS,
                        rescale_scale=VIDEO_RESCALE_SCALE,
                        modality_scale=MODALITY_SCALE,
                        skip_step=0,
                    ),
                    negative.video_encoding,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    MultiModalGuiderParams(
                        cfg_scale=request.audio_cfg_scale,
                        stg_scale=AUDIO_STG_SCALE,
                        stg_blocks=STG_BLOCKS,
                        rescale_scale=AUDIO_RESCALE_SCALE,
                        modality_scale=MODALITY_SCALE,
                        skip_step=0,
                    ),
                    negative.audio_encoding,
                ),
            ),
            sigmas=self.scheduler.execute(steps=request.num_inference_steps).to(dtype=torch.float32, device=p.device),
            noiser=noiser,
            width=shape.width,
            height=shape.height,
            frames=frames,
            fps=FRAME_RATE,
            video=ModalitySpec(context=positive.video_encoding, conditionings=conditionings),
            audio=ModalitySpec(context=positive.audio_encoding),
        )
        synchronize()
        artifact = StageOneArtifact(
            ModelKind.DEV,
            video_state.latent.detach().cpu(),
            audio_state.latent.detach().cpu(),
            generator.get_state().detach().cpu(),
            request.prompt,
            request.negative_prompt,
            seed,
            request.resolution,
            frames,
            FRAME_RATE,
            request.num_inference_steps,
            request.video_cfg_scale,
            request.audio_cfg_scale,
            request.keyframes,
        )
        return (
            p.video_decoder(video_state.latent, None, generator),
            p.audio_decoder(audio_state.latent),
            artifact,
            time.perf_counter() - prompt_started,
            time.perf_counter() - sample_started,
        )

    @torch.inference_mode()
    def production(self, artifact: StageOneArtifact) -> tuple[Iterator[torch.Tensor], Audio, float, float]:
        if artifact.model is not ModelKind.DEV:
            raise ValueError("a dev production requires a quality-preview artifact")
        p = self.pipeline
        started = time.perf_counter()
        generator = torch.Generator(device=p.device).manual_seed(artifact.seed)
        generator.set_state(artifact.generator_state)
        noiser = GaussianNoiser(generator=generator)
        positive, _ = p.prompt_encoder(
            [artifact.prompt, artifact.negative_prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=artifact.seed,
        )
        upscaled = p.upsampler(artifact.video_latent.to(device=p.device, dtype=p.dtype)[:1])
        sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=p.device)
        images = self._image_inputs(artifact.keyframes)
        conditionings = p.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=artifact.resolution.height,
                width=artifact.resolution.width,
                video_encoder=encoder,
                dtype=p.dtype,
                device=p.device,
            )
        )
        video_state, audio_state = p.stage_2(
            denoiser=SimpleDenoiser(positive.video_encoding, positive.audio_encoding),
            sigmas=sigmas,
            noiser=noiser,
            width=artifact.resolution.width,
            height=artifact.resolution.height,
            frames=artifact.num_frames,
            fps=artifact.frame_rate,
            video=ModalitySpec(
                context=positive.video_encoding,
                conditionings=conditionings,
                noise_scale=sigmas[0].item(),
                initial_latent=upscaled,
            ),
            audio=ModalitySpec(
                context=positive.audio_encoding,
                noise_scale=sigmas[0].item(),
                initial_latent=artifact.audio_latent.to(device=p.device, dtype=p.dtype),
            ),
        )
        synchronize()
        return (
            p.video_decoder(video_state.latent, None, generator),
            p.audio_decoder(audio_state.latent),
            0.0,
            time.perf_counter() - started,
        )
