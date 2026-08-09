# ruff: noqa: T201
"""Resumable LTX-API renderer for the first episode of *Deep Space*.

Examples:
    uv run python data/gen.py --scene 1 --stage full --keyframe-root data/keyframes
    uv run python data/gen.py --all --stage full --keyframe-root data/keyframes
    uv run python data/gen.py --scene 1 --video-prompt-file data/ltx_2_3_video_prompts.txt --dry-run
    uv run python data/gen.py --all --stage full --dry-run

Each scene writes independently under ``data/generated_recut_v1/scene_XX``. A stage is
only skipped when both its success flag and its expected output file exist.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from ltx_api import QualityPreviewBuilder, VideoCreator
from ltx_api.types import EnhanceResult, PreviewResult, ProductionResult
from ltx_pipelines.utils.types import OffloadMode

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = REPOSITORY_ROOT / "models"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "generated_recut_v1"
DEFAULT_KEYFRAME_ROOT = DATA_ROOT / "keyframes"
DEFAULT_VIDEO_PROMPT_FILE = DATA_ROOT / "ltx_2_3_video_prompts.txt"
RESOLUTION = (1280, 768)
BASE_SEED = 42_000
QUALITY_PRESETS = ("fast", "standard", "high")
PROMPT_HEADER = re.compile(r"^=== SCENE (?P<number>\d{2}) \| (?P<duration>\d+(?:\.\d+)?) \| (?P<title>.+) ===$")


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


@dataclass(frozen=True)
class ScenePrompt:
    number: int
    title: str
    duration_seconds: float
    prompt: str

    @property
    def full_prompt(self) -> str:
        return self.prompt


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
    result.add_argument(
        "--video-prompt-file",
        type=Path,
        default=DEFAULT_VIDEO_PROMPT_FILE,
        help="LTX I2V prompt text file, split by === SCENE NN | seconds | title ===",
    )
    result.add_argument(
        "--keyframe-root",
        type=Path,
        default=DEFAULT_KEYFRAME_ROOT,
        help="scene keyframes: first.png plus optional keyframes.ini and last.png",
    )
    result.add_argument(
        "--quality",
        choices=QUALITY_PRESETS,
        default="fast",
        help="22B dev preview quality preset (default: fast)",
    )
    result.add_argument("--force", action="store_true", help="rerun selected completed stages")
    result.add_argument("--dry-run", action="store_true", help="validate plan and files without loading models")
    return result


def load_scenes(prompt_file: Path) -> tuple[ScenePrompt, ...]:
    try:
        lines = prompt_file.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read video prompt file: {prompt_file}") from error

    scenes: list[ScenePrompt] = []
    header: re.Match[str] | None = None
    body: list[str] = []

    def finish_scene() -> None:
        if header is None:
            return
        prompt = "\n".join(body).strip()
        if not prompt:
            raise ValueError(f"scene {header['number']} has an empty prompt: {prompt_file}")
        scenes.append(ScenePrompt(int(header["number"]), header["title"], float(header["duration"]), prompt))

    for line in lines:
        match = PROMPT_HEADER.fullmatch(line)
        if match is None:
            if header is not None:
                body.append(line)
            continue
        finish_scene()
        header = match
        body = []
    finish_scene()

    if not scenes:
        raise ValueError(f"no scene delimiter found in video prompt file: {prompt_file}")
    numbers = [scene.number for scene in scenes]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"duplicate scene number in video prompt file: {prompt_file}")
    return tuple(scenes)


# Compatibility for existing local tools that import the default storyboards.  The
# editable source of truth remains the plain-text prompt file above, not Python.
SCENES = load_scenes(DEFAULT_VIDEO_PROMPT_FILE)


def target_scenes(args: argparse.Namespace, scenes: tuple[ScenePrompt, ...]) -> tuple[ScenePrompt, ...]:
    if args.all:
        return scenes
    scenes_by_number = {scene.number: scene for scene in scenes}
    if args.scene not in scenes_by_number:
        raise ValueError(f"unknown scene {args.scene}; choose a number from 1 through {len(scenes)}")
    return (scenes_by_number[args.scene],)


def scene_dir(output_root: Path, scene: ScenePrompt) -> Path:
    return output_root / f"scene_{scene.number:02d}"


def status_path(directory: Path) -> Path:
    return directory / "status.json"


@dataclass(frozen=True)
class TimedKeyframe:
    path: Path
    at_seconds: float
    strength: float


@dataclass(frozen=True)
class SceneKeyframes:
    first: Path
    middle: tuple[TimedKeyframe, ...]
    last: Path | None

    def status_value(self) -> dict[str, Any]:
        return {
            "first": keyframe_file_status(self.first),
            "middle": [
                {
                    "path": str(item.path),
                    "sha256": keyframe_file_status(item.path)["sha256"],
                    "at_seconds": item.at_seconds,
                    "strength": item.strength,
                }
                for item in self.middle
            ],
            "last": keyframe_file_status(self.last) if self.last is not None else None,
        }


def keyframe_file_status(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def scene_keyframes(keyframe_root: Path, scene: ScenePrompt) -> SceneKeyframes:
    directory = keyframe_root / f"scene_{scene.number:02d}"
    first = directory / "first.png"
    last = directory / "last.png"
    if not first.is_file():
        raise FileNotFoundError(f"scene {scene.number:02d} requires an approved first frame: {first}")

    manifest_path = directory / "keyframes.ini"
    if not manifest_path.is_file():
        middle = directory / "middle.png"
        middle_frames = (TimedKeyframe(middle, scene.duration_seconds / 2, 1.0),) if middle.is_file() else ()
        return SceneKeyframes(first, middle_frames, last if last.is_file() else None)

    try:
        manifest = configparser.ConfigParser()
        with manifest_path.open(encoding="utf-8") as stream:
            manifest.read_file(stream)
    except configparser.Error as error:
        raise ValueError(f"invalid keyframe manifest: {manifest_path}") from error

    middle_frames: list[TimedKeyframe] = []
    middle_sections = sorted(section for section in manifest.sections() if section.startswith("middle_"))
    for number, section in enumerate(middle_sections, start=1):
        item = manifest[section]
        relative_path = item.get("file")
        try:
            at_seconds = item.getfloat("at_seconds")
            strength = item.getfloat("strength", fallback=1.0)
        except ValueError as error:
            raise ValueError(f"{section} has an invalid numeric value: {manifest_path}") from error
        if not relative_path:
            raise ValueError(f"{section}.file must be a non-empty path: {manifest_path}")
        path = directory / relative_path
        if not path.is_file():
            continue
        middle_frames.append(TimedKeyframe(path, at_seconds, strength))

    if manifest.has_section("last_frame"):
        relative_path = manifest["last_frame"].get("file")
        if not relative_path:
            raise ValueError(f"last_frame.file must be a path: {manifest_path}")
        last = directory / relative_path
    return SceneKeyframes(first, tuple(middle_frames), last if last.is_file() else None)


def new_status(scene: ScenePrompt, seed: int, quality: str, keyframes: SceneKeyframes) -> dict[str, Any]:
    return {
        "scene": scene.number,
        "title": scene.title,
        "duration_seconds": scene.duration_seconds,
        "resolution": {"width": RESOLUTION[0], "height": RESOLUTION[1]},
        "seed": seed,
        "model": "22b-dev",
        "quality": quality,
        "keyframes": keyframes.status_value(),
        "prompt": scene.full_prompt,
        "prompt_summary": scene.prompt[:180],
        "stages": {},
        "updated_at": timestamp(),
    }


def load_status(
    directory: Path,
    scene: ScenePrompt,
    seed: int,
    quality: str,
    keyframes: SceneKeyframes,
    *,
    reset_preview: bool,
) -> dict[str, Any]:
    path = status_path(directory)
    if not path.exists():
        return new_status(scene, seed, quality, keyframes)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid status file: {path}") from error
    if (
        payload.get("scene") != scene.number
        or payload.get("prompt") != scene.full_prompt
        or payload.get("quality") != quality
        or payload.get("keyframes") != keyframes.status_value()
    ):
        if reset_preview:
            return new_status(scene, seed, quality, keyframes)
        raise ValueError(
            f"status does not match current prompt, quality, or keyframes: {path}; "
            "use --force with --stage preview/full or choose a new --output-root"
        )
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
    quality = getattr(args, "quality", "fast")
    keyframes = scene_keyframes(args.keyframe_root, scene)
    stages = requested_stages(Stage(args.stage))
    status = load_status(
        directory,
        scene,
        seed,
        quality,
        keyframes,
        reset_preview=args.force and Stage.PREVIEW in stages,
    )
    stage_names = ",".join(stage.value for stage in stages)
    logger.write(
        f"scene={scene.number:02d} title={scene.title!r} stages={stage_names} seed={seed} quality={quality} "
        f"first_frame={keyframes.first} middle_frames={len(keyframes.middle)}"
    )

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
                request_builder = (
                    QualityPreviewBuilder()
                    .prompt(scene.full_prompt)
                    .duration_seconds(scene.duration_seconds)
                    .resolution(*RESOLUTION)
                    .seed(seed)
                    .quality(quality)
                    .first_frame(keyframes.first)
                )
                for keyframe in keyframes.middle:
                    request_builder.middle_frame(
                        keyframe.path, at_seconds=keyframe.at_seconds, strength=keyframe.strength
                    )
                if keyframes.last is not None:
                    request_builder.last_frame(keyframes.last)
                request = request_builder.build()
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
        scenes = target_scenes(args, load_scenes(args.video_prompt_file))
    except ValueError as error:
        parser().error(str(error))
    missing = validate_model_root(args.model_root)
    if missing:
        print("Missing required model paths:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2
    print(
        f"[{timestamp()}] planned scenes={len(scenes)} stage={args.stage} quality={args.quality} "
        f"output_root={args.output_root} keyframe_root={args.keyframe_root}",
        flush=True,
    )
    creator = None if args.dry_run else VideoCreator(args.model_root, offload_mode=OffloadMode.DISK)
    success = True
    for scene in scenes:
        success = render_scene(args, scene, creator) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
