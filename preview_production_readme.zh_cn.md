# LTX-2 预览—生产视频验证脚本

`preview_production.py` 是位于仓库根目录的独立本地验证脚本。它不修改 `ltx_pipelines` 包中的现有 Pipeline、CLI 或导出；通过复用 `DistilledPipeline` 和 `ICLoraPipeline` 的已有组件，提供三段式工作流：

```text
prompt + 时长
  └─ fast ──→ preview.mp4 + preview.pt（Stage 1 latent 工件）
                           └─ production ──→ production.mp4

preview.mp4 + 新 prompt
  └─ modify ──→ modified.mp4
```

## 环境准备

从仓库根目录安装锁定依赖：

```bash
uv sync --frozen
```

脚本通过工作区中的 `ltx_pipelines` 和 `ltx_core` 包运行：

```bash
uv run python preview_production.py --help
```

## 模型目录

所有模型路径均相对 `--model-root` 固定，不读取环境变量。当前脚本期望如下布局：

```text
/path/to/models/
├── ltx-2.3-22b-distilled-1.1.safetensors
├── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
├── ltx-2.3-22b-ic-lora-union-control.safetensors
└── gemma/
    └── # Gemma 3 文本编码器的完整下载内容
```

请确认 IC-LoRA 的下载文件名。若实际文件名不同，需要修改 `preview_production.py` 中的 `IC_LORA_RELATIVE_PATH` 常量；其余调用参数不需要携带模型路径或推理配置。

## 1. 生成快速预览

`fast` 仅执行 Distilled Stage 1：在 `384×640` 解码 preview，同时保存用于后续 Stage 2 的 CPU latent artifact。artifact 包含 prompt、seed、最终目标尺寸、帧数、帧率和版本信息。

```bash
uv run python preview_production.py fast \
  --model-root /path/to/models \
  --prompt "A red fox runs through a snowy forest, cinematic tracking shot" \
  --duration-seconds 5 \
  --artifact-path outputs/preview.pt \
  --output-path outputs/preview.mp4 \
  --seed 42
```

`--seed` 可选；未提供时使用 `10`。

## 2. 生成生产视频

`production` 读取 `fast` 保存的 artifact，只执行 Spatial Upscaler 和 Distilled Stage 2，不会重新运行 Stage 1。生产分辨率固定为 `768×1280`，帧率固定为 `24 FPS`。

```bash
uv run python preview_production.py production \
  --model-root /path/to/models \
  --artifact-path outputs/preview.pt \
  --output-path outputs/production.mp4
```

artifact 使用 `torch.save` 保存，但加载时启用 `weights_only=True`，并强制在 CPU 恢复后才移动到当前 Pipeline 设备。它可以被复制到另一进程或另一台机器继续执行；两端应使用兼容的模型版本与本脚本版本。

## 3. 基于 preview 修改视频

`modify` 把 preview 视频作为现有 `ICLoraPipeline` 的 video conditioning 输入，并固定执行 Stage 2。因此输出为生产分辨率，而不是内部低清预览。

```bash
uv run python preview_production.py modify \
  --model-root /path/to/models \
  --preview-video-path outputs/preview.mp4 \
  --prompt "The same fox pauses, looks into the camera, then walks away" \
  --duration-seconds 5 \
  --output-path outputs/modified.mp4 \
  --seed 42
```

## 时长与帧数

脚本固定 `24 FPS`，并沿用仓库已有的 LTX VAE 时间维度规则：帧数必须满足 `8k + 1`。`duration_seconds` 会先换算为帧数，再向下对齐到最近的有效帧数；过短（不足 9 帧）的时长会被拒绝。

例如，5 秒请求会先得到 120 帧，再对齐为 113 帧。因此输出的实际时长可能略短于请求时长。

## 注意事项

- `preview.pt` 是私有 latent 工件，不应作为不可信输入，也不适合公开分发。
- 首次运行会加载大型 checkpoint，并需要与目标分辨率相匹配的 GPU 显存。
- 该脚本是本地验证入口，尚未接入外部 `aideo-models/runtime`。
- 当前仓库未提供上游测试目录；脚本已经过 Ruff、静态编译和 diff 格式检查。请在具备模型与 GPU 的环境中执行上述命令完成端到端验证。
