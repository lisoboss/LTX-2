# ruff: noqa: T201
"""Resumable LTX-API renderer for the first episode of *Deep Space*.

Examples:
    uv run python data/gen.py --scene 1 --stage full
    uv run python data/gen.py --all --stage full
    uv run python data/gen.py --all --stage full --dry-run

Each scene writes independently under ``data/generated_recut_v1/scene_XX``. A stage is
only skipped when both its success flag and its expected output file exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from ltx_api import QualityPreviewBuilder, VideoCreator
from ltx_api.types import EnhanceResult, PreviewResult, ProductionResult
from prompts_en import SCENES, SCENES_BY_NUMBER, ScenePrompt

from ltx_pipelines.utils.types import OffloadMode

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = REPOSITORY_ROOT / "models"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "generated_recut_v1"
RESOLUTION = (1280, 768)
BASE_SEED = 42_000


class Stage(str, Enum):
    PREVIEW = "preview"
    PRODUCTION = "production"
    ENHANCE = "enhance"
    FULL = "full"


STAGE_FILES = {
    Stage.PREVIEW: ("preview.pt", "preview.mp4"),
    Stage.PRODUCTION: ("production.mp4",),
    Stage.ENHANCE: ("enhanced.mp4",),
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SceneLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{line}\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    selection = result.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", type=int, help="one storyboard number, 1 through 15")
    selection.add_argument("--all", action="store_true", help="render all 15 storyboards in number order")
    result.add_argument("--stage", choices=list(Stage), default=Stage.FULL)
    result.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--force", action="store_true", help="rerun selected completed stages")
    result.add_argument("--dry-run", action="store_true", help="validate plan and files without loading models")
    return result


def target_scenes(args: argparse.Namespace) -> tuple[ScenePrompt, ...]:
    if args.all:
        return SCENES
    try:
        return (SCENES_BY_NUMBER[args.scene],)
    except KeyError as error:
        raise ValueError(f"unknown scene {args.scene}; choose a number from 1 through {len(SCENES)}") from error


def scene_dir(output_root: Path, scene: ScenePrompt) -> Path:
    return output_root / f"scene_{scene.number:02d}"


def status_path(directory: Path) -> Path:
    return directory / "status.json"


def new_status(scene: ScenePrompt, seed: int) -> dict[str, Any]:
    return {
        "scene": scene.number,
        "title": scene.title,
        "duration_seconds": scene.duration_seconds,
        "resolution": {"width": RESOLUTION[0], "height": RESOLUTION[1]},
        "seed": seed,
        "model": "22b-dev",
        "quality": "standard",
        "prompt": scene.full_prompt,
        "prompt_summary": scene.prompt[:180],
        "stages": {},
        "updated_at": timestamp(),
    }


def load_status(directory: Path, scene: ScenePrompt, seed: int) -> dict[str, Any]:
    path = status_path(directory)
    if not path.exists():
        return new_status(scene, seed)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid status file: {path}") from error
    if payload.get("scene") != scene.number or payload.get("prompt") != scene.full_prompt:
        raise ValueError(f"status does not match current scene prompt: {path}; choose a new --output-root")
    return payload


def save_status(directory: Path, status: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = timestamp()
    path = status_path(directory)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def outputs_exist(directory: Path, stage: Stage) -> bool:
    return all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in STAGE_FILES[stage])


def completed(status: dict[str, Any], directory: Path, stage: Stage) -> bool:
    return status.get("stages", {}).get(stage.value, {}).get("success") is True and outputs_exist(directory, stage)


def requested_stages(stage: Stage) -> tuple[Stage, ...]:
    if stage is Stage.FULL:
        return (Stage.PREVIEW, Stage.PRODUCTION, Stage.ENHANCE)
    return (stage,)


def invalidate_downstream(status: dict[str, Any], stage: Stage) -> None:
    stages = status.setdefault("stages", {})
    if stage is Stage.PREVIEW:
        stages.pop(Stage.PRODUCTION.value, None)
        stages.pop(Stage.ENHANCE.value, None)
    elif stage is Stage.PRODUCTION:
        stages.pop(Stage.ENHANCE.value, None)


def validate_model_root(model_root: Path) -> list[Path]:
    expected = (
        model_root / "gemma-3-12b-it-qat-q4_0-unquantized",
        model_root / "LTX-2.3/ltx-2.3-22b-dev.safetensors",
        model_root / "LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        model_root / "LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        model_root / "LoRA/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    )
    return [path for path in expected if not path.exists()]


def result_metrics(result: PreviewResult | ProductionResult | EnhanceResult) -> dict[str, Any]:
    return asdict(result.metrics)


def mark_failure(status: dict[str, Any], directory: Path, stage: Stage, error: BaseException) -> None:
    status.setdefault("stages", {})[stage.value] = {
        "success": False,
        "failed_at": timestamp(),
        "error": f"{type(error).__name__}: {error}",
    }
    save_status(directory, status)


def render_scene(args: argparse.Namespace, scene: ScenePrompt, creator: VideoCreator | None) -> bool:
    directory = scene_dir(args.output_root, scene)
    logger = SceneLogger(directory / "workflow.log")
    seed = BASE_SEED + scene.number
    status = load_status(directory, scene, seed)
    stages = requested_stages(Stage(args.stage))
    stage_names = ",".join(stage.value for stage in stages)
    logger.write(f"scene={scene.number:02d} title={scene.title!r} stages={stage_names} seed={seed}")

    for stage in stages:
        if completed(status, directory, stage) and not args.force:
            logger.write(f"stage={stage.value} skipped: success flag and expected output files exist")
            continue
        if args.force:
            invalidate_downstream(status, stage)
        if args.dry_run:
            logger.write(f"stage={stage.value} dry-run: would render {directory}")
            continue
        if creator is None:
            raise RuntimeError("creator is required when --dry-run is not set")
        try:
            logger.write(f"stage={stage.value} started")
            if stage is Stage.PREVIEW:
                request = (
                    QualityPreviewBuilder()
                    .prompt(scene.full_prompt)
                    .duration_seconds(scene.duration_seconds)
                    .resolution(*RESOLUTION)
                    .seed(seed)
                    .quality("standard")
                    .build()
                )
                result = creator.preview(
                    request,
                    artifact_path=directory / "preview.pt",
                    output_path=directory / "preview.mp4",
                )
            elif stage is Stage.PRODUCTION:
                artifact_path = directory / "preview.pt"
                if not artifact_path.is_file():
                    raise FileNotFoundError(f"production requires preview artifact: {artifact_path}")
                result = creator.create_production(
                    artifact_path=artifact_path,
                    output_path=directory / "production.mp4",
                )
            else:
                production_path = directory / "production.mp4"
                if not production_path.is_file():
                    raise FileNotFoundError(f"enhance requires production video: {production_path}")
                result = creator.enhance(
                    production_path=production_path,
                    output_path=directory / "enhanced.mp4",
                    prompt=scene.full_prompt,
                    duration_seconds=scene.duration_seconds,
                    seed=seed,
                )
            status.setdefault("stages", {})[stage.value] = {
                "success": True,
                "completed_at": timestamp(),
                "outputs": [str(directory / name) for name in STAGE_FILES[stage]],
                "metrics": result_metrics(result),
            }
            save_status(directory, status)
            logger.write(f"stage={stage.value} completed")
        except BaseException as error:
            mark_failure(status, directory, stage, error)
            logger.write(f"stage={stage.value} failed: {type(error).__name__}: {error}")
            logger.write(traceback.format_exc().rstrip())
            return False
    return True


def main() -> int:
    args = parser().parse_args()
    try:
        scenes = target_scenes(args)
    except ValueError as error:
        parser().error(str(error))
    missing = validate_model_root(args.model_root)
    if missing:
        print("Missing required model paths:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2
    print(f"[{timestamp()}] planned scenes={len(scenes)} stage={args.stage} output_root={args.output_root}", flush=True)
    creator = None if args.dry_run else VideoCreator(args.model_root, offload_mode=OffloadMode.DISK)
    success = True
    for scene in scenes:
        success = render_scene(args, scene, creator) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
