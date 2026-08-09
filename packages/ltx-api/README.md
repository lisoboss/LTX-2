# ltx-api

`ltx-api` provides a persistent LTX-2.3 workflow: preview → production → enhance.
For image-to-video, create and approve the opening image before building the request.
The opening image fixes frame zero; optional middle and final images guide their
respective frames in both LTX stages.

```python
from pathlib import Path

from ltx_api import QualityPreviewBuilder, VideoCreator

request = (
    QualityPreviewBuilder()
    .prompt("A 35mm camera follows the fox as it runs through falling snow. Wind and pawsteps are audible.")
    .duration_seconds(5)
    .resolution(1280, 768)
    .seed(42)
    .quality("fast")
    .first_frame(Path("frames/opening.png"))
    .middle_frame(Path("frames/turn.png"), at_seconds=2.5)
    .last_frame(Path("frames/ending.png"))
    .build()
)

creator = VideoCreator(Path("models"))
creator.preview(
    request,
    artifact_path=Path("output/preview.pt"),
    output_path=Path("output/preview.mp4"),
)
creator.create_production(
    artifact_path=Path("output/preview.pt"),
    output_path=Path("output/production.mp4"),
)
```

`first_frame()` targets frame 0. `middle_frame()` accepts seconds from the
start and converts them to LTX's 24 fps frame index. `last_frame()` targets the
last valid output frame. Every image path is checked before models are loaded.
The Stage 1 artifact records all resolved keyframes, so production applies the
same conditioning without resampling Stage 1.

`FastPreviewBuilder` supports the same keyframe methods for the distilled
model. Image generation is intentionally outside this package: the caller
controls the approved local image paths.
