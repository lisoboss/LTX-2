"""Run one custom LTX-2.3 prompt through the preview-production workflow.

Examples:
    uv run python data/prompt_demo.py --prompt "A 35mm camera follows a fox through snow..."
    uv run python data/prompt_demo.py --prompt "..." --stage full --quality standard
    uv run python data/prompt_demo.py --prompt-file data/my_prompt.txt --stage enhance

The default ``preview`` stage creates both an MP4 and a reusable Stage 1 latent
artifact.  Use ``--stage production`` or ``--stage enhance`` later without
sampling Stage 1 again.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path

from ltx_api import QualityPreviewBuilder, VideoCreator
from ltx_pipelines.utils.types import OffloadMode


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ROOT = REPOSITORY_ROOT / "models"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "prompt_demo_output"
QUALITY_PRESETS = ("fast", "standard", "high")


class Stage(StrEnum):
    PREVIEW = "preview"
    PRODUCTION = "production"
    ENHANCE = "enhance"
    FULL = "full"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="one complete English LTX prompt")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 text file containing one complete prompt")
    parser.add_argument("--stage", choices=tuple(Stage), default=Stage.PREVIEW)
    parser.add_argument("--quality", choices=QUALITY_PRESETS, default="fast")
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def read_prompt(args: argparse.Namespace) -> str:
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    return prompt


def main() -> int:
    args = build_parser().parse_args()
    try:
        prompt = read_prompt(args)
    except (OSError, ValueError) as error:
        build_parser().error(str(error))

    output = args.output_directory
    artifact_path = output / "preview.pt"
    preview_path = output / "preview.mp4"
    production_path = output / "production.mp4"
    enhanced_path = output / "enhanced.mp4"
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"stage={args.stage} quality={args.quality} duration={args.duration_seconds}s "
        f"resolution={args.width}x{args.height} seed={args.seed} output={output}",
        flush=True,
    )
    creator = VideoCreator(args.model_root, offload_mode=OffloadMode.DISK)
    stage = Stage(args.stage)

    if stage in (Stage.PREVIEW, Stage.FULL):
        request = (
            QualityPreviewBuilder()
            .prompt(prompt)
            .duration_seconds(args.duration_seconds)
            .resolution(args.width, args.height)
            .seed(args.seed)
            .quality(args.quality)
            .build()
        )
        creator.preview(request, artifact_path=artifact_path, output_path=preview_path)
        print(f"preview: {preview_path}\nartifact: {artifact_path}", flush=True)

    if stage in (Stage.PRODUCTION, Stage.FULL):
        if not artifact_path.is_file():
            raise FileNotFoundError(f"production requires preview artifact: {artifact_path}")
        creator.create_production(artifact_path=artifact_path, output_path=production_path)
        print(f"production: {production_path}", flush=True)

    if stage in (Stage.ENHANCE, Stage.FULL):
        if not production_path.is_file():
            raise FileNotFoundError(f"enhance requires production video: {production_path}")
        creator.enhance(
            production_path=production_path,
            output_path=enhanced_path,
            prompt=prompt,
            duration_seconds=args.duration_seconds,
            seed=args.seed,
        )
        print(f"enhanced: {enhanced_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
