"""Static acceptance tests for the 15-shot Deep Space recut."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prompts_en import SCENES


class RecutPromptTests(unittest.TestCase):
    def test_recut_has_fifteen_ordered_shots_totaling_102_seconds(self) -> None:
        assert [scene.number for scene in SCENES] == list(range(1, 16))
        assert len(SCENES) == 15
        assert sum(scene.duration_seconds for scene in SCENES) == 102.0
        assert all(4.0 <= scene.duration_seconds <= 10.0 for scene in SCENES)

    def test_every_prompt_is_a_director_style_audio_visual_instruction(self) -> None:
        for scene in SCENES:
            prompt = scene.prompt.lower()
            with self.subTest(scene=scene.number):
                assert "camera" in prompt
                assert "light" in prompt
                assert "sound" in prompt
                assert "ends" in prompt


if __name__ == "__main__":
    unittest.main()
