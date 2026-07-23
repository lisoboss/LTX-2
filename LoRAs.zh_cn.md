# LTX-2 LoRAs 功能说明

本文仅说明与当前 `LTX-2.3 22B` checkpoint 匹配的 LoRA。LoRA 是附加到基础模型的轻量权重；使用前仍应在模型卡中核对基础模型版本。

## 两种控制方式

| 类型 | 工作方式 | 典型用途 |
| --- | --- | --- |
| **IC-LoRA**（In-Context） | 除文字 prompt 外，还读取参考图片、视频、深度图或姿态等条件。 | 保留主体、参考动作、按控制视频重生成。 |
| **普通 LoRA** | 只以文字 prompt 和 LoRA 权重调节生成。 | 指定镜头运动，例如推近、平移。 |

本仓库的 `preview_production.py enhance` 使用的是 IC-LoRA：把已完成的 `production.mp4` 作为 `video_conditioning` 传入，再执行完整的两阶段生产。因此它适合以已确认的高清成片为基线，做受控优化或变化生成。

## LoRA 一览

| LoRA | 对应模型 | 类型 | 作用 | 适合的输入与场景 |
| --- | --- | --- | --- | --- |
| `LTX-2.3-22b-IC-LoRA-Union-Control` | LTX-2.3 22B | IC-LoRA | 通用参考控制。根据参考视频/图像的视觉与时序信息约束生成。 | 当前工作流使用它；适合“沿用 preview 的主体、构图、动作，再用新 prompt 微调”的场景。 |
| `LTX-2.3-22b-IC-LoRA-Motion-Track-Control` | LTX-2.3 22B | IC-LoRA | 跟随运动轨迹或运动控制信号。 | 需要让角色、物体或镜头遵从给定的移动路径时使用。 |
| `LTX-2.3-22b-IC-LoRA-HDR` | LTX-2.3 22B | 专用 IC-LoRA | HDR 生成/重建控制。 | 需使用仓库的 `HDRICLoraPipeline`，并配套其预计算文本嵌入；不适用于当前通用 `enhance` 命令。 |
| `LTX-2.3-22b-IC-LoRA-LipDub` | LTX-2.3 22B | 专用 IC-LoRA | 口型与音频对齐。 | 需使用 `LipDubPipeline` 和音频条件；不适用于当前通用 `enhance` 命令。 |

## 当前工作流使用的 Union Control

当前已下载并配置的文件为：

```text
models/LoRA/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
```

它由 `preview_production.py` 的 `enhance` 子命令自动加载。流程为：

```text
production.mp4 + 优化 prompt
        │
        ▼
Union-Control IC-LoRA（video_conditioning）
        │
        ▼
Stage 1 受控生成 → 2× 上采样 → Stage 2 精修 → enhanced.mp4
```

`ref0.5` 表示该公开权重的参考控制版本标识。实际控制强度还会受参考视频内容、prompt 与推理参数共同影响；若希望最大程度保留 preview，应保持 prompt 的主体、场景和镜头描述连续，只改需要变化的动作或细节。

## 下载来源

- [Union Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)
- [Motion Track Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control)
- [HDR IC-LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR)
- [LipDub IC-LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub)

下载前请核对 checkpoint 的参数规模与版本；兼容性不明时，优先查阅该 LoRA 模型卡的基础模型声明。
