# LTX-2 预览—生产视频验证脚本

`preview_production.py` 是位于仓库根目录的独立本地验证脚本。它不修改 `ltx_pipelines` 包中的现有 Pipeline、CLI 或导出；通过复用 `DistilledPipeline` 和 `ICLoraPipeline` 的已有组件，提供三段式工作流：

```text
prompt + 时长
  └─ fast ──→ preview.mp4 + preview.pt（Stage 1 latent 工件）
                           └─ production ──→ production.mp4

preview.mp4 + 新 prompt
  └─ enhance ──→ enhanced.mp4
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

脚本默认使用 `--offload disk`：权重按需从磁盘读取，只保留少量 pinned CPU buffer，适用于 12 GB 显存和有限系统内存，但速度最慢。`--offload cpu` 会把完整权重预加载到 pinned 系统内存（通常约需 36 GB）；只有显存约 28 GB 或更高时才使用 `--offload none`。

## 模型目录

所有模型路径均相对 `--model-root` 固定，不读取环境变量。当前脚本期望如下布局：

```text
/path/to/models/
├── LTX-2.3/
│   ├── ltx-2.3-22b-distilled-1.1.safetensors
│   ├── ltx-2.3-22b-distilled-lora-384-1.1.safetensors
│   ├── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
├── LoRA/
│   └── ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
└── gemma-3-12b-it-qat-q4_0-unquantized/
    └── # Gemma 3 文本编码器的完整下载内容
```

IC-LoRA 固定使用 `LoRA/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`；其余调用参数不需要携带模型路径或推理配置。

## 1. 生成快速预览

`fast` 仅执行 Distilled Stage 1：在 `384×640` 解码 preview，同时保存用于后续 Stage 2 的 CPU latent artifact。为减少前期试错成本，fast 使用固定 4 步 preview sigma schedule（生产仍使用完整的 Stage 2 精修 schedule）。artifact 包含 prompt、seed、Stage 1 结束时的随机生成器状态、最终目标尺寸、帧数、帧率和版本信息。

```bash
uv run python preview_production.py fast \
  --model-root /path/to/models \
  --prompt "A red fox runs through a snowy forest, cinematic tracking shot" \
  --duration-seconds 5 \
  --artifact-path outputs/preview.pt \
  --offload disk \
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
  --offload disk \
  --output-path outputs/production.mp4
```

artifact 使用 `torch.save` 保存，但加载时启用 `weights_only=True`，并强制在 CPU 恢复后才移动到当前 Pipeline 设备。它可以被复制到另一进程或另一台机器继续执行；两端应使用兼容的模型版本与本脚本版本。artifact 还保存 Stage 1 完成时的 generator state，Stage 2 恢复它后可沿用原版一次性两阶段流程的随机序列。

## 3. 基于 preview 修改视频

`enhance` 把 `production.mp4` 作为现有 `ICLoraPipeline` 的 video conditioning 输入，并固定执行 Stage 2。因此输出为生产分辨率，而不是内部低清预览。它是基于成片的受控重新生成，不是无损的像素级后处理。

```bash
uv run python preview_production.py enhance \
  --model-root /path/to/models \
  --production-video-path outputs/production.mp4 \
  --prompt "The same fox pauses, looks into the camera, then walks away" \
  --duration-seconds 5 \
  --offload disk \
  --output-path outputs/enhanced.mp4 \
  --seed 42
```

## 一次执行完整工作流（可恢复）

`run_preview_production_workflow.sh` 依次执行上述三个阶段。每个阶段完成后在输出目录写入一个 `.done` 成功标志；重复执行时，标志和输出文件都存在的阶段会跳过。fast 重新执行时会清除 production 与 enhance 的成功标志。

```bash
./run_preview_production_workflow.sh
```

默认参数写在 Bash 脚本顶部：模型目录 `./models`、时长 5 秒、seed 42、`disk` offload，以及两个 fox prompt。按需直接修改对应变量。输出目录包含 `preview.mp4`、`preview.pt`、`production.mp4`、`enhanced.mp4`，以及 `fast.done`、`production.done`、`enhance.done`。命令失败不会写成功标志；下次运行会从失败阶段继续。删除某个 `.done` 文件可强制重跑该阶段。

## 时长与帧数

脚本固定 `24 FPS`，并沿用仓库已有的 LTX VAE 时间维度规则：帧数必须满足 `8k + 1`。`duration_seconds` 会先换算为帧数，再向下对齐到最近的有效帧数；过短（不足 9 帧）的时长会被拒绝。

例如，5 秒请求会先得到 120 帧，再对齐为 113 帧。因此输出的实际时长可能略短于请求时长。

## 注意事项

- `preview.pt` 是私有 latent 工件，不应作为不可信输入，也不适合公开分发。
- 首次运行会加载大型 checkpoint，并需要与目标分辨率相匹配的 GPU 显存。
- 该脚本是本地验证入口，尚未接入外部 `aideo-models/runtime`。
- 当前仓库未提供上游测试目录；脚本已经过 Ruff、静态编译和 diff 格式检查。请在具备模型与 GPU 的环境中执行上述命令完成端到端验证。
