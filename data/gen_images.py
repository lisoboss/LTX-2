"""Generate selectable LTX I2V keyframe candidates through LAN ComfyUI.

Examples:
    uv run python data/gen_images.py --scene 1 --frame first --dry-run
    uv run python data/gen_images.py --scene 1 --frame first  # 10 random candidates
    uv run python data/gen_images.py --all --frame all

The source prompts are the independent Chinese image prompts in
``data/keyframes/提示词.md``.  Candidates are kept separate from the approved
``data/keyframes/scene_XX`` images that LTX consumes.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = DATA_ROOT / "keyframes" / "提示词.md"
DEFAULT_WORKFLOW_FILE = DATA_ROOT / "workflows" / "comfui-gen-image-api.json"
DEFAULT_CANDIDATE_ROOT = DATA_ROOT / "keyframe_candidates"
DEFAULT_COMFY_URL = "http://192.168.31.3:8000"
TARGET_SIZE = (1280, 768)

# Nodes in the exported, versioned ``comfui-gen-image-api.json`` workflow.
POSITIVE_PROMPT_NODE = "82"
SEED_NODE = "17"
FINAL_SAVE_NODE = "118"  # The second, latent-upscaled sampling pass.
PREVIEW_PREFIX_NODE = "117:119"
FINAL_PREFIX_NODE = "131:126"

SCENE_HEADER = re.compile(r"^## 分镜 (?P<number>\d+)：")
FRAME_HEADER = re.compile(
    r"^`(?P<filename>(?:first|middle_\d+|last)\.png)\s·\s(?P<seconds>\d+(?:\.\d+)?)s`(?P<note>.*)$"
)


@dataclass(frozen=True)
class ImagePrompt:
    scene: int
    filename: str
    at_seconds: float
    prompt: str | None
    solid_color: tuple[int, int, int] | None


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", type=int, help="generate one storyboard scene")
    selection.add_argument("--all", action="store_true", help="generate every scene in the prompt file")
    parser.add_argument("--frame", choices=("first", "middle", "last", "all"), default="all")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--workflow-file", type=Path, default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
        help="unapproved candidate images; these are separate from data/keyframes",
    )
    parser.add_argument("--samples", type=int, default=10, help="random candidates per prompt (default: 10)")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="ComfyUI seed; -1 lets the rgthree Seed node choose a fresh random seed each run",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--force", action="store_true", help="regenerate images that already exist")
    parser.add_argument("--dry-run", action="store_true", help="validate the plan without contacting ComfyUI")
    return parser.parse_args()


def prompt_color(note: str) -> tuple[int, int, int] | None:
    if "纯黑" in note:
        return (0, 0, 0)
    if "纯蓝紫" in note:
        return (45, 20, 90)
    return None


def load_prompts(path: Path) -> tuple[ImagePrompt, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts: list[ImagePrompt] = []
    scene: int | None = None
    index = 0
    while index < len(lines):
        scene_match = SCENE_HEADER.fullmatch(lines[index])
        if scene_match is not None:
            scene = int(scene_match["number"])
            index += 1
            continue
        frame_match = FRAME_HEADER.fullmatch(lines[index])
        if frame_match is None or scene is None:
            index += 1
            continue
        filename = frame_match["filename"]
        at_seconds = float(frame_match["seconds"])
        solid_color = prompt_color(frame_match["note"])
        index += 1
        while index < len(lines) and not lines[index].startswith("```text"):
            if SCENE_HEADER.fullmatch(lines[index]) or FRAME_HEADER.fullmatch(lines[index]):
                break
            index += 1
        if index >= len(lines) or not lines[index].startswith("```text"):
            if solid_color is None:
                raise ValueError(f"missing text prompt for scene {scene:02d} {filename}: {path}")
            prompts.append(ImagePrompt(scene, filename, at_seconds, None, solid_color))
            continue
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].startswith("```"):
            body.append(lines[index])
            index += 1
        if index == len(lines):
            raise ValueError(f"unclosed prompt block for scene {scene:02d} {filename}: {path}")
        prompt = "\n".join(body).strip()
        if not prompt:
            raise ValueError(f"empty prompt for scene {scene:02d} {filename}: {path}")
        prompts.append(ImagePrompt(scene, filename, at_seconds, prompt, solid_color))
        index += 1
    if not prompts:
        raise ValueError(f"no keyframe prompts found: {path}")
    return tuple(prompts)


def selected(prompts: tuple[ImagePrompt, ...], args: argparse.Namespace) -> tuple[ImagePrompt, ...]:
    result = tuple(item for item in prompts if args.all or item.scene == args.scene)
    if args.frame != "all":
        result = tuple(item for item in result if item.filename.startswith(args.frame))
    if not result:
        raise ValueError("the requested scene/frame has no image prompt")
    return result


def request_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 -- URL is explicit CLI input.
        return json.load(response)


def read_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 -- URL is explicit CLI input.
        return json.load(response)


def read_bytes(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:  # noqa: S310 -- URL is explicit CLI input.
        return response.read()


def require_workflow_nodes(workflow: dict[str, Any]) -> None:
    for node in (POSITIVE_PROMPT_NODE, SEED_NODE, FINAL_SAVE_NODE, PREVIEW_PREFIX_NODE, FINAL_PREFIX_NODE):
        if node not in workflow:
            raise ValueError(f"workflow does not match the ComfyUI image adapter; missing node {node}")


def output_path(root: Path, item: ImagePrompt) -> Path:
    return root / f"scene_{item.scene:02d}" / Path(item.filename).stem


def status_path(root: Path, item: ImagePrompt) -> Path:
    return root / f"scene_{item.scene:02d}" / "image_status.json"


def update_status(root: Path, item: ImagePrompt, payload: dict[str, Any]) -> None:
    path = status_path(root, item)
    status: dict[str, Any] = {}
    if path.is_file():
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    candidate = str(payload.get("candidate", "confirmed"))
    frame_status = status.setdefault(item.filename, {})
    if not isinstance(frame_status, dict):
        frame_status = {}
        status[item.filename] = frame_status
    frame_status[candidate] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_for_ltx(image_bytes: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.png")
    with Image.open(io.BytesIO(image_bytes)) as source:
        source = source.convert("RGB")
        source_ratio = source.width / source.height
        target_ratio = TARGET_SIZE[0] / TARGET_SIZE[1]
        if source_ratio > target_ratio:
            width = round(source.height * target_ratio)
            left = (source.width - width) // 2
            source = source.crop((left, 0, left + width, source.height))
        elif source_ratio < target_ratio:
            height = round(source.width / target_ratio)
            top = (source.height - height) // 2
            source = source.crop((0, top, source.width, top + height))
        source.resize(TARGET_SIZE, Image.Resampling.LANCZOS).save(temporary, format="PNG")
    temporary.replace(target)


def make_solid_image(color: tuple[int, int, int], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", TARGET_SIZE, color).save(target, format="PNG")


def submit_image(base_url: str, workflow: dict[str, Any], item: ImagePrompt, seed: int) -> tuple[str, dict[str, Any]]:
    prompt = copy.deepcopy(workflow)
    prefix = f"codex_keyframes/scene_{item.scene:02d}/{Path(item.filename).stem}"
    prompt[POSITIVE_PROMPT_NODE]["inputs"]["text"] = item.prompt
    prompt[SEED_NODE]["inputs"]["seed"] = seed
    prompt[PREVIEW_PREFIX_NODE]["inputs"]["value"] = prefix
    prompt[FINAL_PREFIX_NODE]["inputs"]["value"] = prefix
    response = request_json(f"{base_url}/prompt", {"prompt": prompt, "client_id": "ltx-keyframe-generator"})
    if response.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected the workflow: {response['node_errors']}")
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str):
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {response}")
    return prompt_id, prompt


def await_final_image(base_url: str, prompt_id: str, poll_seconds: float) -> dict[str, Any]:
    while True:
        history = read_json(f"{base_url}/history/{prompt_id}")
        record = history.get(prompt_id)
        if record is not None:
            status = record.get("status", {})
            if status.get("completed"):
                outputs = record.get("outputs", {}).get(FINAL_SAVE_NODE, {}).get("images", [])
                if not outputs:
                    raise RuntimeError(f"ComfyUI job completed without final image output: {prompt_id}")
                return outputs[0]
            raise RuntimeError(f"ComfyUI job ended without success: {status}")
        time.sleep(poll_seconds)


def image_url(base_url: str, result: dict[str, Any]) -> str:
    required = ("filename", "subfolder", "type")
    if any(not isinstance(result.get(key), str) for key in required):
        raise RuntimeError(f"invalid ComfyUI image result: {result}")
    return f"{base_url}/view?{urlencode({key: result[key] for key in required})}"


def main() -> int:
    args = parse_arguments()
    try:
        if args.samples < 1:
            raise ValueError("--samples must be at least 1")
        prompts = selected(load_prompts(args.prompt_file), args)
        workflow = json.loads(args.workflow_file.read_text(encoding="utf-8"))
        require_workflow_nodes(workflow)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for item in prompts:
        samples = 1 if item.solid_color is not None else args.samples
        for sample in range(1, samples + 1):
            target = output_path(args.output_root, item) / f"candidate_{sample:02d}.png"
            label = f"scene={item.scene:02d} frame={item.filename} candidate={sample:02d} seed={args.seed}"
            if target.is_file() and target.stat().st_size > 0 and not args.force:
                print(f"[{timestamp()}] {label} skipped: {target} already exists", flush=True)
                continue
            if args.dry_run:
                kind = "solid color" if item.solid_color is not None else "ComfyUI generation"
                print(f"[{timestamp()}] {label} dry-run: {kind} -> {target}", flush=True)
                continue
            try:
                print(f"[{timestamp()}] {label} started", flush=True)
                if item.solid_color is not None:
                    make_solid_image(item.solid_color, target)
                    prompt_id = None
                    result = None
                else:
                    prompt_id, _ = submit_image(args.comfy_url.rstrip("/"), workflow, item, args.seed)
                    print(f"[{timestamp()}] {label} prompt_id={prompt_id}", flush=True)
                    result = await_final_image(args.comfy_url.rstrip("/"), prompt_id, args.poll_seconds)
                    fit_for_ltx(read_bytes(image_url(args.comfy_url.rstrip("/"), result)), target)
                update_status(
                    args.output_root,
                    item,
                    {
                        "success": True,
                        "candidate": f"candidate_{sample:02d}",
                        "completed_at": timestamp(),
                        "requested_seed": args.seed,
                        "prompt": item.prompt,
                        "prompt_id": prompt_id,
                        "comfy_output": result,
                        "output": str(target),
                        "resolution": {"width": TARGET_SIZE[0], "height": TARGET_SIZE[1]},
                    },
                )
                print(f"[{timestamp()}] {label} completed: {target}", flush=True)
            except (HTTPError, URLError, OSError, RuntimeError, ValueError) as error:
                update_status(
                    args.output_root,
                    item,
                    {
                        "success": False,
                        "candidate": f"candidate_{sample:02d}",
                        "failed_at": timestamp(),
                        "requested_seed": args.seed,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                print(f"[{timestamp()}] {label} failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
