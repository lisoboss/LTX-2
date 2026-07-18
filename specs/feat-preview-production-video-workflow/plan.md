# Technical Plan: feat-preview-production-video-workflow

## Directory structure

```text
project/
├── preview_production.py              # [new] standalone preview/production test script
├── packages/ltx-pipelines/
├── tests/
│   ├── test_distilled_stages.py     # [new] artifact, stage isolation, persistence tests
│   ├── test_ic_lora_preview.py      # [new] preview conditioning and production default tests
│   └── test_functional_models.py    # [new] fixed model-root wrappers and duration alignment tests
```

`aideo-models/runtime` is not present in this repository. The public wrappers above are the integration boundary it can instantiate using only `model_root`; wiring that external runtime remains a downstream change.

## Core data model

```python
@dataclass
class DistilledStage1Artifact:
    version: int
    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    prompt: str
    seed: int
    height: int                 # final, stage-2 target height
    width: int                  # final, stage-2 target width
    num_frames: int
    frame_rate: float

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "DistilledStage1Artifact": ...
```

Persistence stores a CPU-only tensor payload plus serializable metadata. `load()` validates the schema/version; `run_stage_2()` performs pipeline-specific semantic checks and moves tensors to the current pipeline device without mutating the caller's artifact.

```python
@dataclass
class FastVideoResult:
    video: Iterator[torch.Tensor]
    audio: Audio
    artifact: DistilledStage1Artifact
```

## Interfaces

```python
class PreviewProductionPipeline:
    def run_stage_1(
        self, *, prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float, images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None, enhance_prompt: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
    ) -> FastVideoResult: ...

    def run_stage_2(
        self, artifact: DistilledStage1Artifact, *,
        images: list[ImageConditioningInput] | None = None,
        tiling_config: TilingConfig | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]: ...
```

`PreviewProductionPipeline` is a new adapter around an injected `DistilledPipeline`. It calls the existing pipeline blocks in the same order but exposes the two stage boundaries. Prompt context is recomputed in stage 2 from artifact metadata rather than serialized, so artifacts never depend on process- or device-local encoder state. The existing `DistilledPipeline.__call__()` is not changed.

```python
class PreviewVideoModifyPipeline:
    def modify_preview(
        self, *, preview_video_path: str | Path, prompt: str, seed: int,
        height: int, width: int, num_frames: int, frame_rate: float,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]: ...
```

`PreviewVideoModifyPipeline` wraps an injected `ICLoraPipeline`; `modify_preview()` delegates to its established `__call__()` path with the preview as IC-LoRA `video_conditioning` and `skip_stage_2=False`. It does not duplicate conditioning or diffusion logic, and `ICLoraPipeline` itself is not changed.

```python
class LTX2FastVideo:
    def __init__(self, model_root: Path): ...
    def generate(self, *, prompt: str, duration_seconds: float, seed: int | None = None) -> FastVideoResult: ...

class LTX2HighResolutionVideo:
    def __init__(self, model_root: Path): ...
    def generate(self, *, artifact: DistilledStage1Artifact) -> tuple[Iterator[torch.Tensor], Audio]: ...

class LTX2VideoModify:
    def __init__(self, model_root: Path): ...
    def generate(self, *, preview_video_path: str | Path, prompt: str,
                 duration_seconds: float, seed: int | None = None) -> tuple[Iterator[torch.Tensor], Audio]: ...
```

The wrappers own fixed feature constants: model-relative checkpoint/Gemma/upsampler/LoRA locations, preview and production dimensions, frame rate, seeds, sigma schedules, and local lifecycle. The duration rule reuses the repository's established VAE rule: derive the requested frame count from duration and FPS, then snap **down** to `8k + 1` using `SpatioTemporalScaleFactors.default().time` (the same implementation as `lipdub._snap_frames_to_8k1`). The public helper and tests will make rounding behavior explicit. The unresolved IC-LoRA filename remains an explicit constant to confirm against the downloaded model layout.

## Validation and error handling

- Reject unsupported artifact versions, non-CPU-safe/malformed tensor payloads, missing video/audio latents, non-finite fps, frame counts not satisfying `8k + 1`, and latent batch/temporal/spatial shapes inconsistent with the stored target geometry.
- Require artifact dimensions and frame rate to match the production wrapper's fixed configuration before stage 2.
- Keep stage-1 prompt/image conditioning semantics intact; stage-2 image conditioning remains optional and uses current behavior.
- Do not modify existing pipeline source, exports, CLI, or documentation. The standalone script imports the installed `ltx_pipelines` package directly.

## Implementation phases

### Phase 1 — Artifact and staged adapter

- Add the artifact/result model and CPU serialization contract.
- Implement a new staged adapter that orchestrates existing `DistilledPipeline` components without altering their diffusion algorithms.
- Add fake-block tests proving stage 1 does not instantiate/call upsampling or stage 2 and that stage 2 does not call stage 1.

### Phase 2 — Preview modification and feature wrappers

- Add a narrow `ICLoraPipeline` adapter with production stage 2 enabled.
- Add model-root functional wrappers and fixed configuration constants.
- Add duration-to-frame alignment tests and wrapper construction tests that verify no caller model configuration is accepted.

### Phase 3 — Compatibility verification

- Test artifact save/load across fresh pipeline instances and full legacy `__call__()` behavior.
- Run the new module's package tests/lint/type checks available in this checkout.
