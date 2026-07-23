# LTX Pipelines：成片逻辑说明

本文基于以下源码的实际调用顺序，说明“成片”如何产生，以及它们与本仓库 `fast / production / enhance` 工作流的关系：

- `ltx_pipelines.distilled`
- `ltx_pipelines.t2a_one_stage`
- `ltx_pipelines.ti2vid_two_stages`
- `ltx_pipelines.ti2vid_two_stages_hq`

## 先说结论

对于视频 pipeline，**成片不是单纯把低清视频放大**。它是：先在低分辨率 latent 空间决定内容、动作和镜头，再把 latent 上采样，并在目标分辨率进行第二次扩散精修，最后由 VAE 解码、封装音频，得到 MP4。

```text
prompt / 图片条件 / seed
            │
            ▼
文本编码、条件编码、随机噪声
            │
            ▼
Stage 1：低分辨率扩散，确定主要内容与时序
            │
            ▼
latent 上采样（不是像素放大）
            │
            ▼
Stage 2：目标分辨率扩散精修，补充细节并保持时序
            │
            ▼
VAE 解码视频 + 音频解码 + MP4 编码 = 成片
```

这里的 Stage 2 会再次执行扩散去噪，因此它不仅提升分辨率，也会改变和完善纹理、细节、局部运动与音画结果。

## 四个上游 pipeline 的区别

| Pipeline | 输入 | Stage 1 | Stage 2 | 输出与定位 |
| --- | --- | --- | --- | --- |
| `DistilledPipeline` | prompt，可选图片 | distilled 扩散，半目标分辨率 | 上采样 latent 后，distilled 扩散精修 | 当前预览—生产工作流的基础；适合快速、高效的文本/图像转视频。 |
| `TI2VidTwoStagesPipeline` | prompt、negative prompt、可选图片、CFG 参数 | 完整模型 + CFG，在半目标分辨率生成 | 上采样后用 distilled LoRA 精修 | 更可控的通用文/图生视频；Stage 1 通常比 distilled 更重。 |
| `TI2VidTwoStagesHQPipeline` | 同上，另有 HQ LoRA 强度 | 完整模型 + CFG + `res_2s` 二阶采样 | 上采样后以 `res_2s` 精修 | 面向更高质量/更少采样步数的两阶段视频生成。 |
| `T2AOneStagePipeline` | prompt、negative prompt、音频 CFG 参数 | 仅音频的一阶段扩散 | 无 | 只生成音频文件，不生成视频成片。 |

## 1. `DistilledPipeline`：当前工作流的原始成片路径

`distilled.py` 的完整调用顺序是：

1. 用 `seed` 创建同一个随机生成器和噪声器。
2. `PromptEncoder` 把 prompt 编码为视频、音频 context。
3. Stage 1 在 `width / 2 × height / 2` 的 latent 空间去噪，输出 `video_state.latent` 与 `audio_state.latent`。
4. `VideoUpsampler` 将 Stage 1 的视频 latent 放大到目标分辨率。
5. Stage 2 以这个上采样 latent 为初始状态，在全分辨率继续去噪；音频 latent 也继续精修。
6. `VideoDecoder`、`AudioDecoder` 解码，`encode_video` 将二者写入 MP4。

因此，原版 `DistilledPipeline` 的“成片”就是 **Stage 1 + latent 上采样 + Stage 2 + 解码/封装**。它不在 Stage 1 输出时单独保存可复用工件。

## 2. 当前 `fast / production / enhance` 如何拆开原版逻辑

| 文件 | 实际逻辑 | 是否重跑 Stage 1 | 与原版成片的关系 |
| --- | --- | --- | --- |
| `preview.mp4`（`fast`） | 以较短的 4 步 Stage 1 生成低分辨率视频并解码；同时保存 latent artifact。 | 是 | 用来判断动作、镜头、prompt 和 seed，不是最终质量的原版成片。 |
| `production.mp4`（`production`） | 读取 artifact 中的 Stage 1 latent，执行上采样、Stage 2、解码。 | 否 | 最接近“把确认过的 fast 直接完成成片”的路径。 |
| `enhanced.mp4`（`enhance`） | 读取 `production.mp4` 作为 IC-LoRA 视频条件，重新执行受控的两阶段生成。 | 是 | 是基于已完成成片的受控优化候选，不是同一 latent 的直接续算。 |

为了使拆分后的 `production` 延续 Stage 1 的随机过程，artifact 同时保存 Stage 1 结束时的随机生成器状态；生产阶段恢复该状态后再进入 Stage 2。

### 应该保留哪个成片？

- 只希望把已确认 preview 高清化：选 `production.mp4`。
- 希望以 production 为基线优化动作、主体细节或 prompt：选 `enhanced.mp4`。
- 最佳实践：保留 `preview.mp4`、artifact、`production.mp4` 和候选 `enhanced.mp4`；这样既能追溯，也能避免重跑昂贵的 Stage 1。

## 3. `TI2VidTwoStagesPipeline`：完整模型决定内容，distilled LoRA 完成精修

它同样是两阶段，但 Stage 1 与 `DistilledPipeline` 不同：

- Stage 1 使用完整模型的 scheduler 和 CFG（正向/负向 prompt）。
- 可使用文字以外的图片条件。
- Stage 2 将用户 LoRA 与 distilled LoRA 一起加载，以较短的蒸馏精修步骤完成目标分辨率成片。

它的核心分工是：完整模型负责前期的可控内容生成；蒸馏 Stage 2 负责较快完成高分辨率细化。

## 4. `TI2VidTwoStagesHQPipeline`：二阶采样的高质量两阶段路径

HQ pipeline 保持与上一节相同的两阶段结构，但有两个主要变化：

- 使用 `res_2s` 二阶采样循环，而非普通的一阶采样方式；目标是在较少步数下保持可比质量。
- distilled LoRA 在 Stage 1、Stage 2 的强度可分别设置，便于平衡生成特性与精修强度。

它依然遵循“低分辨率决定结构 → latent 上采样 → 高分辨率精修 → 解码”的成片逻辑。

## 5. `T2AOneStagePipeline`：不参与视频成片

`t2a_one_stage.py` 只构建音频相关模型权重，并向 `DiffusionStage` 传入 `video=None`。它在单次扩散后直接解码音频并调用 `encode_audio` 写入文件。

它可以用作独立配音、音效或音频生成工具，但不包含视频 Stage 1、上采样或 Stage 2，不能替代任一视频成片 pipeline。

## 对资源和迭代速度的含义

```text
fast：     低分辨率 Stage 1（4 步）+ 低清解码
production：                 latent 上采样 + Stage 2 + 高清解码
enhance：  production 成片编码 + 受控 Stage 1 + 上采样 + Stage 2 + 高清解码
```

所以当前拆分的价值在于：修改 prompt、seed 或镜头意图时，只花较低成本反复生成 `fast`；确认后只执行一次 `production`。只有确实要让 IC-LoRA 根据 production 成片和优化 prompt 重新解释内容时，才运行成本更高的 `enhance`。
