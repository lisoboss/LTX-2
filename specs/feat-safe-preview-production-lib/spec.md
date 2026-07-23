# feat-safe-preview-production-lib

## 背景与目标

为本地 LTX-2.3 工作流提供一个根目录 Python 库，统一构造 preview、production、enhance 三阶段，并只向普通调用者暴露安全、可解释的生成参数。

## 用户故事

- 作为工作流调用者，我希望以 `22b-dev` 或 `22b-distilled` 生成 preview，以便低成本确定 prompt、seed、动作与镜头。
- 作为工作流调用者，我希望将 preview artifact 续算为 production，以便不重复 Stage 1。
- 作为工作流调用者，我希望以 production.mp4 为 IC-LoRA 参考生成 enhanced.mp4，以便基于确认的成片做受控优化。
- 作为高级调用者，我希望绕过质量档位直接指定步数与 CFG，以便精确控制质量与耗时。

## 功能列表

### MVP（必须有）

- [ ] 新增 `packages/ltx-api` Python 包，不修改现有 `ltx-core` 与 `ltx-pipelines` 的上游实现。
- [ ] 提供 `preview`、`production`、`enhance` 三阶段 Python API。
- [ ] `preview` 支持模型选择：`dev`、`distilled`。
- [ ] `production` 不重复对应模型的 Stage 1，并生成 `production.mp4`。
- [ ] `enhance` 固定使用 distilled + Union Control IC-LoRA，以 `production.mp4` 为视频条件并生成 `enhanced.mp4`。
- [ ] 对外公开：`prompt`、`negative_prompt`、`seed`、`num_inference_steps`、`video_cfg_scale`、`audio_cfg_scale`、`duration`、`resolution`。
- [ ] 当模型为 `dev` 且未提供专家覆盖值时，质量档位映射为：
  - `fast`：12 steps、video CFG 2.5；
  - `standard`：30 steps、video CFG 3.0；
  - `high`：40 steps、video CFG 3.0。
- [ ] 所有未公开的采样、STG、rescale、跨模态 guidance、Stage 2 schedule、LoRA 强度保持内部固定。
- [ ] 保存可跨进程加载的 Stage 1 artifact，包含 continuation 所需 latent、生成器状态和元数据。
- [ ] 每个阶段 API 负责原子写入 MP4；返回结果路径与性能元数据，不向 MVP 调用者暴露惰性 video/audio 迭代器。
- [ ] distilled 模型收到非默认 `negative_prompt`、`video_cfg_scale` 或 `audio_cfg_scale` 时显式报错。
- [ ] 所有输入与输出 `Path` 均由外部调用者提供；库不得生成默认输出目录、文件名或缓存位置。
- [ ] `VideoResolution` 在构造时强制校验为正整数，且宽高均为 64 的倍数；错误必须在模型加载前报告。
- [ ] `VideoResolution` 校验失败时，错误信息必须包含根据当前输入向上取整得到的最小合法 `VideoResolution(width=..., height=...)` 建议。
- [ ] 两个 preview request builder 均提供默认 `duration_seconds=5.0`、最终 `resolution=1280×768`（preview 为 640×384）、`seed=42`，但 `prompt` 必须显式设置。
- [ ] dev builder 默认 `quality=standard`，解析为 30 steps、video CFG 3.0、audio CFG 7.0；negative prompt 默认空字符串。

### 后续迭代（可以有）

- [ ] CLI 包装、批处理接口和进度回调。
- [ ] 可选的质量档位自定义与配置文件。
- [ ] IC-LoRA conditioning 强度作为专家参数。

## 技术方案概述

- 平台/运行环境：Python 3.12、PyTorch、现有 `ltx_core` 与 `ltx_pipelines`。
- 库位置：`packages/ltx-api/src/ltx_api/`，独立包但依赖现有 `ltx_core` 与 `ltx_pipelines`。
- 模块划分：公共类型与安全参数、质量档位解析、artifact 持久化、dev adapter、distilled adapter、IC-LoRA enhance adapter、对外 workflow facade 分别放在独立模块。
- `dev`：按 `TI2VidTwoStagesPipeline` 的全模型 Stage 1（官方 2.3 guider 参数）与 Stage 2 逻辑拆分并持久化 artifact。
- `distilled`：复用现有 root `preview_production.py` 的 Stage 1 artifact / Stage 2 adapter。
- `enhance`：复用现有 root `preview_production.py` 的 `LTX2VideoEnhance`，内部固定 distilled IC-LoRA pipeline。

## 非功能需求

- 性能：production 必须复用 artifact，不得重跑 Stage 1。
- 资源：默认 `disk` offload，允许库构造时设置但不作为普通生成参数。
- 安全：artifact 使用 CPU tensors 和 `torch.load(..., weights_only=True)`。
- 兼容性：dev 和 distilled artifact 必须带模型类型与版本，禁止交叉续算。

## 已确认的 API 边界

- Python API 负责 MP4 与 artifact 落盘；MVP 不返回需要调用者自行编码的惰性 media 结果。
- distilled 模型不支持 negative prompt 与 CFG；调用者传入非默认值时显式报错。
- 外部调用者完全控制 artifact、preview MP4、production MP4 与 enhanced MP4 的路径；结果对象只回显实际写入路径和元数据。
