# feat-image-guided-video-api

## 背景与目标

为 `ltx-api` 增加基于已确认关键帧图片的图生视频能力，替代 prompt demo 的纯文生视频路径。调用方先生成并挑选图片，再由 LTX 生成运动、镜头与同步音频。

## 用户故事

- 作为视频创作者，我希望把首帧图片传给 LTX，以便锁定人物、场景和开场构图。
- 作为视频创作者，我希望可选地传入中间帧和尾帧，以便约束事件节点与结尾画面。
- 作为工作流调用者，我希望 production 从 artifact 恢复相同关键帧条件，以便不重跑 Stage 1 且不丢失图片约束。

## MVP

- `FastPreviewBuilder` 和 `QualityPreviewBuilder` 均提供 `first_frame()`、`middle_frame()`、`last_frame()`。
- 公共 API 只接收本地 `Path`、时间（秒）和强度，不暴露上游 `ImageConditioningInput`。
- 首帧目标为 frame 0；中间帧由 24 fps 换算为帧号；尾帧目标为最后一帧。
- builder 在模型加载前校验图片存在、时间范围、强度和重复目标帧。
- dev / distilled 的 Stage 1 与 Stage 2 均应用关键帧条件。
- artifact 以可安全加载的基础数据持久化关键帧；旧 artifact 没有关键帧时仍可继续 production。
- `data/prompt_demo.py` 的 preview/full 模式要求 `--first-frame`，可重复 `--middle-frame IMAGE SECONDS`，可选 `--last-frame`。
- `data/gen.py` 从 `scene_XX/first.png` 读取每镜首帧；Python 3.10 标准库可读取的 `keyframes.ini` 可定义任意多个可选 `middle_XX.png` 的准确秒点及可选 `last.png`，批量生成不再静默回退到纯 T2V。

## 非目标

- 不在 `ltx-api` 内生成静态图片；图片生成、审阅与选择由调用者控制。
- 不修改 `ltx-core` 或 `ltx-pipelines` 上游代码。
