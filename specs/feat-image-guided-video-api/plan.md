# Technical Plan: feat-image-guided-video-api

1. 在公开类型中加入 `ImageKeyframe`，并在两种 preview request 中携带已解析的关键帧列表。
2. 在两个 builder 中提供自然语言式的 `first_frame`、`middle_frame`、`last_frame` 方法；内部把秒数换算为 24 fps 帧索引。
3. 扩展 artifact schema，保存基础类型的图片路径、帧号和强度；加载 v1 artifact 时默认没有关键帧。
4. dev 和 distilled adapter 在 Stage 1 与 Stage 2 分别调用现有 `combined_image_conditionings`，不修改上游 pipeline。
5. 将 `data/prompt_demo.py` 改为首帧驱动，提供可重复的中间帧参数与可选尾帧参数；将批量 `data/gen.py` 约定为每镜 `first.png`、由 Python 3.10 标准库可读的 `keyframes.ini` 定义任意多个可选 `middle_XX.png` 与可选 `last.png`。

## 验证边界

不启动模型或 GPU 任务。调用方在本地图片确认后执行 demo；builder 会在模型创建前校验关键帧路径和时间。
