"""Deprecated compatibility view of the editable LTX I2V prompt text.

``data/ltx_2_3_video_prompts.txt`` is the sole prompt source used by
``data/gen.py``.  This module only keeps ``SCENES`` available for older local
tools; it does not assemble a shared style, location, or prompt prefix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_HEADER = re.compile(r"^=== SCENE (?P<number>\d{2}) \| (?P<duration>\d+(?:\.\d+)?) \| (?P<title>.+) ===$")
_SOURCE = Path(__file__).with_name("ltx_2_3_video_prompts.txt")


@dataclass(frozen=True)
class ScenePrompt:
    number: int
    title: str
    duration_seconds: float
    prompt: str

    @property
    def full_prompt(self) -> str:
        return self.prompt


def _load() -> tuple[ScenePrompt, ...]:
    scenes: list[ScenePrompt] = []
    header: re.Match[str] | None = None
    body: list[str] = []

    def finish() -> None:
        if header is not None:
            scenes.append(
                ScenePrompt(int(header["number"]), header["title"], float(header["duration"]), "\n".join(body).strip())
            )

    for line in _SOURCE.read_text(encoding="utf-8").splitlines():
        match = _HEADER.fullmatch(line)
        if match is None:
            if header is not None:
                body.append(line)
            continue
        finish()
        header, body = match, []
    finish()
    return tuple(scenes)


SCENES = _load()
SCENES_BY_NUMBER = {scene.number: scene for scene in SCENES}
