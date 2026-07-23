# `preview_production.py` 中文逐段注释版

这是 [preview_production.py](preview_production.py) 的阅读指南。为避免破坏可运行脚本，本文件按源文件行号解释每一条有效语句的职责；空行仅用于分隔逻辑块。

## 1. 文件头、导入与固定配置（1–44）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 1–6 | 模块 docstring | 说明本文件是对原有 LTX Pipeline 的组合封装，不改包内实现。 |
| 8 | `from __future__ import annotations` | 延迟解析类型注解，使注解可引用后面定义的类。 |
| 10 | `import argparse` | 提供 `fast`、`production`、`modify` 三个 CLI 子命令。 |
| 11 | `Iterator` | 标记视频是惰性帧迭代器，而不是一次性加载的完整视频。 |
| 12 | `dataclass` | 定义 artifact 与快速结果这两个数据容器。 |
| 13 | `Path` | 统一处理模型、artifact 和视频输出路径。 |
| 15 | `torch` | Tensor、随机数生成器、推理模式和序列化都依赖 PyTorch。 |
| 17 | `GaussianNoiser` | 根据 seed 创建扩散采样所需的高斯噪声。 |
| 18–19 | LoRA 类型与重命名映射 | 把磁盘中的 LoRA 权重正确映射到 LTX Transformer 参数名。 |
| 20 | `TilingConfig`、`get_video_chunks_number` | 控制 VAE 解码分块并告知 MP4 编码器视频分块数量。 |
| 21 | `Audio`、`SpatioTemporalScaleFactors` | 前者是音频返回类型，后者提供 LTX 的时间压缩比例。 |
| 22–23 | `DistilledPipeline`、`ICLoraPipeline` | 复用仓库已有的两阶段扩散及 IC-LoRA 实现。 |
| 24 | `ImageConditioningInput` | 保持与原 Pipeline 的图像 conditioning 参数兼容。 |
| 25 | 两套 sigma schedule | Stage 1 用 8 步 distilled schedule；Stage 2 用短精修 schedule。 |
| 26 | `SimpleDenoiser` | distilled 推理使用单次前向 denoiser，不使用 CFG/STG。 |
| 27 | 分辨率与 conditioning helper | 验证二阶段尺寸，并构造图像条件。 |
| 28 | `encode_video` | 把惰性帧和音频写成 MP4。 |
| 29 | `ModalitySpec`、`OffloadMode` | 前者描述视频/音频 latent 输入；后者选择 none/cpu/disk 权重流式模式。 |
| 31 | `ARTIFACT_VERSION` | artifact 格式版本；当前为 2，因为新增了 generator state。旧 artifact 会被明确拒绝。 |
| 32–35 | preview/production 尺寸 | preview 解码为 384×640；artifact 的最终目标是 768×1280。 |
| 36–39 | FPS、seed、offload、preview sigma 默认值 | 固定 24 FPS、seed 10、磁盘流式以适配低显存 GPU；`FAST_PREVIEW_SIGMAS` 每隔一个 sigma 取样，形成 4 步预览 schedule。 |
| 40–44 | 模型相对路径常量 | 所有路径以 `--model-root` 为根；包括 checkpoint、upscaler、distilled LoRA、Gemma 与 IC-LoRA。 |

## 2. 时长、帧数与 artifact（47–124）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 47 | `duration_to_num_frames` | 将用户秒数转换为 LTX 可接受的帧数。 |
| 49–52 | `torch.isfinite` 与正数检查 | 拒绝 `NaN`、无穷、0 或负时长/FPS。 |
| 54 | `round(duration * fps)` | 将请求时长换算为最近整数帧。 |
| 55 | `time_scale` | 读取 VAE 默认时间尺度，当前为 8。 |
| 56 | `((n - 1)//scale)*scale + 1` | 向下对齐为 `8k+1` 帧，满足因果时间 VAE。 |
| 57–58 | 最小帧数检查 | 少于一个时间组时给出明确错误。 |
| 62 | `DistilledStage1Artifact` | 定义可跨进程保存的 Stage 1 状态。 |
| 66–68 | `video_latent`、`audio_latent`、`generator_state` | 保存 Stage 1 两种 latent，以及 Stage 1 结束时的随机数状态，供 Stage 2 继续。 |
| 68–73 | prompt、seed、尺寸、帧数、FPS | 保存重建 Stage 2 prompt context 和验证兼容性所需元数据。 |
| 74 | `version` | 默认写入当前 artifact 格式版本。 |
| 76–77 | `__post_init__` | 每次新建或加载 artifact 都立即验证。 |
| 79–94 | `save` | 创建父目录，将所有 tensor 拷到 CPU，再写入同目录临时文件。 |
| 81–82 | `Path` 与 `mkdir` | 允许 `outputs/...` 这类尚不存在的目录。 |
| 83–94 | `payload` | 保存安全的标量、字符串和 CPU contiguous tensor，包括 generator state。 |
| 94 | `torch.save` | 写入临时 artifact，而非直接覆盖最终 artifact。 |
| 95–97 | `replace`/`finally` | 成功时原子替换；失败时删除临时文件。 |
| 100–115 | `load` | 用 `weights_only=True` 在 CPU 加载，检查 schema 后重建 artifact。 |
| 102–112 | `required` | 明确允许的字段集合，防止缺失或多余字段悄悄通过。 |
| 113–114 | keys 严格比较 | 不支持的 artifact 格式会报错，而不是生成错误视频。 |
| 118–124 | `FastVideoResult` | 把 preview 视频 iterator、音频和 artifact 作为一次 fast 调用的结果返回。 |

## 3. Stage 1：只采样低清预览（127–196）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 127–131 | `PreviewProductionPipeline` | 适配器保存一个现有 `DistilledPipeline` 实例。 |
| 133 | `@torch.inference_mode()` | Stage 1 主体不构建反向图，减少显存。 |
| 134–147 | `run_stage_1` 参数 | 接收完整生产目标尺寸，但内部自动用一半分辨率采样。 |
| 149 | `assert_resolution` | 验证目标生产尺寸符合二阶段 LTX 对齐要求。 |
| 150 | `_validate_frame_metadata` | 验证 `8k+1` 和有效 FPS。 |
| 151 | `images = images or []` | 让未给图像条件时使用空列表。 |
| 152–154 | pipeline/generator/noiser | 为本次调用建立可复现的随机噪声源。 |
| 155–159 | `prompt_encoder` | 获取 video/audio 文本 context；可选地把第一张图用于 prompt enhancement。 |
| 160 | `height//2`、`width//2` | Stage 1 仅生成低清 latent。 |
| 161–170 | `image_conditioner` | 临时创建 VAE encoder，并把图像转换成 Stage 1 尺寸的条件。 |
| 171–181 | `pipeline.stage(...)` | 真正执行 Stage 1 扩散；fast 功能层传入 4 步 preview sigma，没有 upsampler、没有 Stage 2 sigma。 |
| 172–180 | denoiser/sigma/noiser/modality | 分别提供文本预测器、8 步 schedule、随机源、视频与音频输入规范。 |
| 182–192 | `DistilledStage1Artifact(...)` | 立即将 latent detach 并转 CPU，同时在 preview 解码前捕获 generator state，保证 Stage 2 可恢复原始随机序列。 |
| 192–196 | `FastVideoResult(...)` | 返回解码视频、解码音频和 artifact。视频用 `_inference_iterator` 包装，解决惰性解码时 inference mode 已退出的问题。 |

## 4. Stage 2：从 artifact 继续，不重跑 Stage 1（198–255）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 198 | `@torch.inference_mode()` | Stage 2 主体关闭梯度。 |
| 199–206 | `run_stage_2` | 只接收 artifact；图像条件和 tiling 可选。 |
| 208 | `_validate_artifact` | 拒绝版本、形状、尺寸或帧数不兼容的 artifact。 |
| 209–217 | pipeline/随机源 | 先以 artifact seed 重建 generator，再用 artifact generator state 覆盖其状态，随后创建 noiser。 |
| 213–217 | 再次 prompt encode | artifact 保存文本而不保存设备相关 context；因此可跨进程恢复。 |
| 218–219 | `.to(device, dtype)` | 只在需要 Stage 2 时把 CPU latent 移回当前 GPU。 |
| 220 | `pipeline.upsampler` | 对 Stage 1 视频 latent 做空间上采样；这正是 Stage 2 的起点。 |
| 221 | Stage 2 sigma | 将短精修 schedule 放到当前设备。 |
| 222–231 | Stage 2 图像条件 | 用生产分辨率重新编码可选图像条件。 |
| 232–251 | 第二次 `pipeline.stage` | 用上采样视频和保存的音频作为 initial latent 做精修。 |
| 240–245 | 视频 `ModalitySpec` | `noise_scale` 取首个 Stage 2 sigma，`initial_latent` 是 upsampled latent。 |
| 246–250 | 音频 `ModalitySpec` | 音频不重新随机初始化，而是从 artifact 音频 latent 精修。 |
| 252–255 | decoder 返回值 | 延迟解码生产视频；同样用 `_inference_iterator` 保持 inference mode。 |

## 5. preview 修改与功能包装（258–346）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 258–262 | `PreviewVideoModifyPipeline` | 包装既有 `ICLoraPipeline`，不复制其 conditioning 实现。 |
| 264 | inference decorator | 修改流程不需要梯度。 |
| 265–276 | `modify_preview` 参数 | 明确传入 preview 路径、新 prompt、生成规格。 |
| 278–289 | `self._pipeline(...)` | 将 preview 作为唯一 `video_conditioning`，强度 1.0，且固定 `skip_stage_2=False`。 |
| 285 | `images=[]` | 修改入口不附加图像条件。 |
| 286 | `video_conditioning` | IC-LoRA 实际读取并编码 preview 视频的位置。 |
| 290 | `_inference_iterator` | 修复 IC-LoRA 惰性 VAE 解码的 inference tensor 错误。 |
| 292–314 | `LTX2FastVideo` | 仅需 model root/offload；`generate` 固定生产目标尺寸，输出其中一半分辨率 preview，并明确传入 4 步 `FAST_PREVIEW_SIGMAS`。 |
| 310–320 | `LTX2HighResolutionVideo` | 加载同一类 Pipeline；先验证 artifact 是固定 768×1280、24 FPS，再执行 Stage 2。 |
| 323–346 | `LTX2VideoModify` | 加载 IC-LoRA Pipeline；把时长转换为帧数后调用修改 adapter。 |

## 6. 模型构造与 artifact 校验（349–394）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 349–362 | `_make_distilled_pipeline` | 根据 `model_root` 拼 checkpoint/Gemma/upscaler 路径，加入已验证的 distilled LoRA，并传入 offload 模式。 |
| 354–360 | `LoraPathStrengthAndSDOps` | 指定 distilled LoRA 路径、强度 1.0 和权重名映射。 |
| 365–373 | `_make_ic_lora_pipeline` | 构造 IC-LoRA Pipeline；此处的 `IC_LORA_RELATIVE_PATH` 文件必须真实存在。 |
| 376–386 | `_validate_artifact` | 检查版本、tensor 类型、维度、batch size、分辨率和时间元数据。 |
| 379–382 | latent 维度检查 | 视频必须是五维 tensor，音频至少三维，防止错误文件进入模型。 |
| 383–384 | batch 检查 | 当前工作流只支持 batch size 1。 |
| 389–394 | `_validate_frame_metadata` | 统一检查 `8k+1` 与有效 FPS。 |

## 7. CLI 入口（397–454）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 397–400 | `main` 与 subparsers | 创建命令行入口和必须选择的子命令。 |
| 402–408 | `fast` 参数 | 要求 prompt、时长、artifact 输出、视频输出；seed 可选。 |
| 410–413 | `production` 参数 | 仅需要 artifact 输入与生产视频输出。 |
| 415–421 | `modify` 参数 | 要求 preview 视频、新 prompt、时长和输出。 |
| 423 | `parse_args` | 将 shell 参数解析成 `args`。 |
| 424–431 | fast 分支 | 调用 Stage 1，再由 `_publish_fast_result` 成对发布 artifact 与 preview。 |
| 432–435 | production 分支 | 加载 artifact，调用 Stage 2，然后原子写入生产视频。 |
| 436–443 | modify 分支 | 调用 IC-LoRA 修改，按相同 duration 规则原子写入 modified 视频。 |
| 446–454 | `_add_model_arguments` | 三个子命令共享 model root 和 offload 参数；默认 `disk`。 |

## 8. 原子输出与惰性解码（457–525）

| 行 | 代码/语句 | 作用 |
| --- | --- | --- |
| 457–471 | `_encode` | 创建输出目录，先编码到临时 MP4，成功后原子替换最终 MP4。 |
| 464 | `mkdir` | 修复 `Parent directory outputs does not exist`。 |
| 465 | `_temporary_path` | 临时文件与目标在同一目录，保证 `replace` 是同文件系统原子操作。 |
| 466–470 | `try/finally` | 成功发布最终文件；失败时清理临时文件。 |
| 473–496 | `_publish_fast_result` | 同时处理 preview 与 artifact：两份临时文件都成功后才发布最终路径。 |
| 481–482 | 两个临时路径 | 避免采样成功、视频编码失败时留下可误用的最终 artifact。 |
| 484 | 临时 artifact | 先保存 CPU latent。 |
| 485–492 | 临时 preview 编码 | 消费惰性视频 iterator 并写入临时 MP4。 |
| 493–494 | 两次 `replace` | 将两个完整文件发布到用户请求的最终路径。 |
| 495–497 | 清理 | 任意异常后删除未发布的临时文件。 |
| 500–513 | `_encode_to_path` | 单纯调用仓库 `encode_video`；由外层决定是否原子发布。 |
| 516–518 | `_temporary_path` | 保留原后缀（如 `.mp4`/`.pt`），让编码器识别格式。 |
| 521–524 | `_inference_iterator` | 在真正执行惰性 VAE decode 时重新进入 inference mode，修复 `Inference tensors do not track version counter`。 |
| 525 | `if __name__ == "__main__"` | 仅在直接运行脚本时调用 CLI；被 Bash workflow 或其他 Python 导入时不自动执行。 |

## 9. 运行时流程小结

1. `fast`：文本 → 低清 Stage 1 latent → CPU artifact + preview MP4。
2. `production`：读取 CPU artifact → spatial upsample → Stage 2 精修 → production MP4。
3. `modify`：preview MP4 → IC-LoRA video conditioning → 两阶段修改 → modified MP4。
4. `run_preview_production_workflow.sh` 通过 `.done` 文件跳过已成功阶段；每条外部命令和进度都写入带时间戳的 `workflow.log`。
