"""No-GPU acceptance tests for the recut generator configuration."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


class FakeBuilder:
    latest: FakeBuilder | None = None

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        type(self).latest = self

    def prompt(self, value: str) -> FakeBuilder:
        self.values["prompt"] = value
        return self

    def duration_seconds(self, value: float) -> FakeBuilder:
        self.values["duration_seconds"] = value
        return self

    def resolution(self, width: int, height: int) -> FakeBuilder:
        self.values["resolution"] = (width, height)
        return self

    def seed(self, value: int) -> FakeBuilder:
        self.values["seed"] = value
        return self

    def quality(self, value: str) -> FakeBuilder:
        self.values["quality"] = value
        return self

    def build(self) -> dict[str, object]:
        return self.values


@dataclass
class FakeMetrics:
    prompt_seconds: float = 0.0
    sample_seconds: float = 0.0
    encode_seconds: float = 0.0
    total_seconds: float = 0.0
    frames: int = 1
    frame_rate: float = 24.0
    output_resolution: object = None


class FakeCreator:
    def preview(self, request: dict[str, object], *, artifact_path: Path, output_path: Path) -> object:
        self.request = request
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"artifact")
        output_path.write_bytes(b"preview")
        return types.SimpleNamespace(metrics=FakeMetrics())


def load_generator() -> types.ModuleType:
    data_directory = Path(__file__).parent
    sys.path.insert(0, str(data_directory))
    ltx_api = types.ModuleType("ltx_api")
    ltx_api.QualityPreviewBuilder = FakeBuilder
    ltx_api.VideoCreator = FakeCreator
    ltx_api_types = types.ModuleType("ltx_api.types")
    ltx_api_types.EnhanceResult = object
    ltx_api_types.PreviewResult = object
    ltx_api_types.ProductionResult = object
    ltx_types = types.ModuleType("ltx_pipelines.utils.types")
    ltx_types.OffloadMode = types.SimpleNamespace(DISK="disk")
    sys.modules.update(
        {
            "ltx_api": ltx_api,
            "ltx_api.types": ltx_api_types,
            "ltx_pipelines": types.ModuleType("ltx_pipelines"),
            "ltx_pipelines.utils": types.ModuleType("ltx_pipelines.utils"),
            "ltx_pipelines.utils.types": ltx_types,
        }
    )
    spec = importlib.util.spec_from_file_location("recut_gen", data_directory / "gen.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GeneratorRecutTests(unittest.TestCase):
    def test_recut_uses_new_output_root_and_fast_preview_request(self) -> None:
        generator = load_generator()
        assert generator.DEFAULT_OUTPUT_ROOT.name == "generated_recut_v1"
        scene = generator.SCENES[0]
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                output_root=Path(temporary),
                stage="preview",
                force=False,
                dry_run=False,
            )
            creator = FakeCreator()
            assert generator.render_scene(args, scene, creator)
        assert creator.request["quality"] == "fast"
        assert FakeBuilder.latest is not None
        assert FakeBuilder.latest.values["quality"] == "fast"


if __name__ == "__main__":
    unittest.main()
