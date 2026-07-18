# feat-preview-production-video-workflow

## 背景与目标

为 Aideo 等上层 Runtime 提供三段式 LTX-2 视频流程：低分辨率预览、基于预览 latent 的生产级高分辨率输出、基于预览的视频修改。现有 `DistilledPipeline` 将 Stage 1、空间提升和 Stage 2 封装为一次调用，无法让确认过的预览复用于后续生产任务。

## 用户故事

- 作为 Runtime 开发者，我希望仅传 `model_root: Path` 创建 LTX 功能模型，模型子路径和推理配置由 LTX-2 固定。
- 作为创作者，我希望用 `prompt + duration_seconds` 快速生成低分辨率预览，先验证内容。
- 作为创作者，我希望将已确认的预览提升为生产视频，而不是重新执行 Stage 1 采样。
- 作为创作者，我希望基于预览视频和新 prompt 修改视频，以保留其主体与运动参考。

## MVP

### 1. Stage 1 工件

- 新增公开、Runtime 无关的 `DistilledStage1Artifact`。
- 保存 Stage 1 视频 latent、音频 latent、prompt、seed、目标尺寸、帧数、帧率和版本信息。
- 提供 CPU 安全的序列化/反序列化接口，支持在后续进程重新加载。
- Pipeline 不处理任务 ID、HTTP、SSE、对象存储或路径策略；调用方决定持久化位置。

### 2. 快速视频

- 在 `DistilledPipeline` 提供公开的 Stage 1 运行接口。
- 只运行低分辨率 Stage 1 和解码；不构造或调用 Spatial Upscaler、Stage 2 diffusion、Stage 2 sigma schedule。
- 请求最小字段：`prompt`、`duration_seconds`，`seed` 可选。
- 返回低分辨率预览视频、音频和 `DistilledStage1Artifact`。

### 3. 高分辨率视频

- 在 `DistilledPipeline` 提供公开的 Stage 2 运行接口。
- 输入为 `DistilledStage1Artifact`，不得重新运行 Stage 1。
- 复用工件中的视频与音频 latent，执行 Spatial Upscaler、Stage 2 精修和解码。
- 必须校验 artifact 版本、latent 形状、帧率和目标分辨率。

### 4. 视频修改

- 为 `ICLoraPipeline` 提供面向预览视频的功能入口。
- 请求最小字段：`preview_video_path`、`prompt`、`duration_seconds`，`seed` 可选。
- 固定 IC-LoRA 相对路径和两阶段配置；构造函数只接收 `model_root`。
- 默认执行 Stage 2；`skip_stage_2` 仅保留为内部预览能力。

### 5. 面向功能的包装类

| 类 | 责任 | 最小请求 | 输出 |
| --- | --- | --- | --- |
| `LTX2FastVideo` | Distilled Stage 1 预览 | `prompt`, `duration_seconds` | preview video + artifact |
| `LTX2HighResolutionVideo` | Stage 2 生产提升 | `artifact` | production video |
| `LTX2VideoModify` | IC-LoRA 修改 | `preview_video_path`, `prompt`, `duration_seconds` | modified video |

包装类只接收 `model_root: Path`，实现本项目的同步或异步本地模型生命周期；不得读取 Aideo Runtime 环境变量。

## 数据流

```text
prompt + duration
        │
        ▼
DistilledPipeline.run_stage_1()
        ├── preview.mp4
        └── Stage1Artifact（私有持久化 latent）
                         │
                         ▼
              DistilledPipeline.run_stage_2()
                         │
                         ▼
              production.mp4

preview.mp4 + edit prompt → ICLoraPipeline.modify_preview() → modified.mp4
```

## 技术约束

- 复用 `DistilledPipeline` 现有 PromptEncoder、DiffusionStage、VideoUpsampler、VideoDecoder 和 AudioDecoder；不得复制推理算法。
- 模型根目录下的 checkpoint、Gemma、Spatial Upscaler、Distilled LoRA 与 IC-LoRA 路径为 LTX-2 功能层常量，不使用环境变量。
- `duration_seconds` 在功能包装层转换为与 LTX 时间尺度对齐的 `num_frames`；对齐规则必须公开并有单元测试。
- artifact tensor 持久化前迁移到 CPU，恢复时移动到 Pipeline device。
- 保持 `DistilledPipeline.__call__()` 和现有 CLI 的兼容性。

## 验收标准

- fake Pipeline 测试证明快速视频只调用 Stage 1，并返回 artifact。
- fake Pipeline 测试证明高分辨率视频只调用 Spatial Upscaler 与 Stage 2，不重新调用 Stage 1。
- artifact 序列化后可在新 Pipeline 实例中恢复并用于 Stage 2。
- 视频修改测试证明预览视频传入 IC-LoRA conditioning，且生产模式默认执行 Stage 2。
- Aideo Runtime 可仅传 `model_root` 创建三项功能；请求不携带模型路径、量化、offload、宽高或 sigma 配置。

## 待定问题

- 固定 IC-LoRA 的实际文件名与相对路径，需要以实际下载的修改/控制 LoRA 确认。
- 快速预览建议固定为 384×640、24 FPS；生产建议固定为 768×1280 或 1088×1920，均待目标 GPU 显存验证。
- Temporal Upscaler 不属于本次 MVP。
