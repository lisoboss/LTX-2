# Tasks: feat-safe-preview-production-lib

> 生成时间：2026-07-23
> 基于：spec.md + plan.md
> 验证环境：`ai.wsl` 的 `/mnt/g/AI/LTX-2`；GPU 实测由该服务器执行。

## 进度

- [ ] 0 / 16 任务完成

---

### Task 1: 创建 `packages/ltx-api/pyproject.toml`

- **文件**：`packages/ltx-api/pyproject.toml`
- **类型**：包配置
- **依赖**：无
- **描述**：定义 `ltx-api` uv package，声明对 `ltx-core`、`ltx-pipelines` 的 workspace 依赖和测试依赖。
- **验收**：`uv sync --frozen` 能发现 package；`uv run python -c "import ltx_api"` 在骨架完成后可执行。

---

### Task 2: 创建 `packages/ltx-api/src/ltx_api/types.py`

- **文件**：`packages/ltx-api/src/ltx_api/types.py`
- **类型**：实现
- **依赖**：Task 1
- **描述**：实现业务化 request/result 类型、质量档位、内部模型类型、指标类型，以及带 64 倍数和最小合法建议错误信息的 `VideoResolution`。
- **验收**：不合法 `VideoResolution(1000, 720)` 的异常包含 `VideoResolution(width=1024, height=768)`。

---

### Task 3: 创建 `packages/ltx-api/tests/test_types.py`

- **文件**：`packages/ltx-api/tests/test_types.py`
- **类型**：测试
- **依赖**：Task 2
- **描述**：覆盖 `VideoResolution` 的合法值、非整数/非正值、64 倍数错误建议与业务结果对象。
- **验收**：`uv run pytest packages/ltx-api/tests/test_types.py` 通过。

---

### Task 4: 创建 `packages/ltx-api/src/ltx_api/presets.py`

- **文件**：`packages/ltx-api/src/ltx_api/presets.py`
- **类型**：实现
- **依赖**：Task 2
- **描述**：实现 `fast / standard / high` 的 dev 档位映射及固定 LTX-2.3 STG、rescale、跨模态 guidance 参数。
- **验收**：standard 映射为 30 / 3.0 / 7.0，专家覆盖独立于映射定义。

---

### Task 5: 创建 `packages/ltx-api/src/ltx_api/builders.py`

- **文件**：`packages/ltx-api/src/ltx_api/builders.py`
- **类型**：实现
- **依赖**：Task 2、Task 4
- **描述**：实现 `FastPreviewBuilder` 与 `QualityPreviewBuilder`；给出 5 秒、1280×768、seed 42 默认值，强制 prompt，并解析质量档位与显式专家覆盖。
- **验收**：fast builder 没有 negative prompt/CFG/steps 方法；quality builder 的覆盖优先于档位。

---

### Task 6: 创建 `packages/ltx-api/tests/test_builders.py`

- **文件**：`packages/ltx-api/tests/test_builders.py`
- **类型**：测试
- **依赖**：Task 5
- **描述**：覆盖两个 builder 默认值、prompt 必填、分辨率校验、档位映射和专家参数优先级。
- **验收**：`uv run pytest packages/ltx-api/tests/test_builders.py` 通过。

---

### Task 7: 创建 `packages/ltx-api/src/ltx_api/artifacts.py`

- **文件**：`packages/ltx-api/src/ltx_api/artifacts.py`
- **类型**：实现
- **依赖**：Task 2
- **描述**：实现版本化 Stage 1 artifact 的 CPU 序列化、`weights_only=True` 加载、原子保存，以及 dev/distilled model-kind 校验。
- **验收**：artifact schema 包含 latents、RNG state、metadata 与 model kind，且跨模型续算会被拒绝。

---

### Task 8: 创建 `packages/ltx-api/tests/test_artifacts.py`

- **文件**：`packages/ltx-api/tests/test_artifacts.py`
- **类型**：测试
- **依赖**：Task 7
- **描述**：使用小型 CPU tensor 覆盖 artifact round-trip、schema 拒绝、原子写入及 model-kind 校验。
- **验收**：`uv run pytest packages/ltx-api/tests/test_artifacts.py` 通过。

---

### Task 9: 创建 `packages/ltx-api/src/ltx_api/media.py`

- **文件**：`packages/ltx-api/src/ltx_api/media.py`
- **类型**：实现
- **依赖**：Task 2
- **描述**：封装惰性 VAE iterator 的 inference-mode 生命周期、原子 MP4 编码、CUDA 计时和 `StageMetrics` 收集。
- **验收**：编码助手保证目标父目录存在，并只在成功后替换最终文件。

---

### Task 10: 创建 `packages/ltx-api/src/ltx_api/distilled.py`

- **文件**：`packages/ltx-api/src/ltx_api/distilled.py`
- **类型**：实现
- **依赖**：Task 2、Task 7、Task 9
- **描述**：使用现有 LTX building blocks 实现 distilled 的可持久化 Stage 1 preview 和不重跑 Stage 1 的 Stage 2 production；拒绝不支持的 negative prompt/CFG 参数。
- **验收**：adapter 不调用上采样/Stage 2 时完成 preview；production 从 artifact continuation 续算。

---

### Task 11: 创建 `packages/ltx-api/src/ltx_api/dev.py`

- **文件**：`packages/ltx-api/src/ltx_api/dev.py`
- **类型**：实现
- **依赖**：Task 2、Task 4、Task 7、Task 9
- **描述**：按 `TI2VidTwoStagesPipeline` 语义拆分完整 dev 的 CFG Stage 1、artifact 保存、latent 上采样与 Stage 2 production，并恢复 RNG state。
- **验收**：dev preview 采用已解析的 quality/专家参数；production 不重跑 Stage 1。

---

### Task 12: 创建 `packages/ltx-api/src/ltx_api/enhance.py`

- **文件**：`packages/ltx-api/src/ltx_api/enhance.py`
- **类型**：实现
- **依赖**：Task 2、Task 9
- **描述**：实现固定 distilled + Union Control IC-LoRA adapter，以外部提供的 production MP4 作为视频条件，输出 enhanced MP4。
- **验收**：固定使用 Union Control 路径与内部参数，不暴露非安全 IC-LoRA 参数。

---

### Task 13: 创建 `packages/ltx-api/src/ltx_api/workflow.py`

- **文件**：`packages/ltx-api/src/ltx_api/workflow.py`
- **类型**：实现
- **依赖**：Task 10、Task 11、Task 12
- **描述**：实现 `VideoCreator.preview`、`create_production`、`enhance` facade；根据 request 类型派发 adapter，所有路径都由外部传入。
- **验收**：返回三个业务 result；production 自动依据 artifact model kind 选择 adapter；API 返回前 MP4/artifact 已存在。

---

### Task 14: 创建 `packages/ltx-api/tests/test_workflow.py`

- **文件**：`packages/ltx-api/tests/test_workflow.py`
- **类型**：测试
- **依赖**：Task 13
- **描述**：通过 mock adapter 覆盖 facade 路由、外部路径透传、错误传播及不允许交叉 artifact 的行为。
- **验收**：`uv run pytest packages/ltx-api/tests/test_workflow.py` 通过，无需 GPU。

---

### Task 15: 创建 `packages/ltx-api/src/ltx_api/__init__.py` 与 `py.typed`

- **文件**：`packages/ltx-api/src/ltx_api/__init__.py`、`packages/ltx-api/src/ltx_api/py.typed`
- **类型**：公开 API
- **依赖**：Task 5、Task 13
- **描述**：仅导出 `VideoCreator`、两个 builder、request/result、`VideoResolution` 与质量档位；不暴露 LTX 内部 adapter；添加空的 `py.typed` PEP 561 标记。
- **验收**：`from ltx_api import VideoCreator, FastPreviewBuilder, QualityPreviewBuilder` 成功，构建产物包含 `ltx_api/py.typed`。

---

### Task 16: 创建 `packages/ltx-api/README.md` 并执行服务器验证

- **文件**：`packages/ltx-api/README.md`
- **类型**：文档与集成验证
- **依赖**：Task 15
- **描述**：记录最小调用、默认值、路径责任、quality/专家覆盖与 enhance 语义；在 `ai.wsl` 运行 `uv sync --frozen`、pytest，并实际执行 distilled fast → production → enhance 的 5 秒 smoke test。dev 路径至少执行 import/builder 验证；若显存与时间允许，再运行 dev fast smoke test。
- **验收**：服务器日志带时间戳；pytest 全绿；distilled 三阶段输出均存在且 result/文件路径一致。
