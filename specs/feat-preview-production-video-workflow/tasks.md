# Tasks: feat-preview-production-video-workflow

> 生成时间：2026-07-16
> 基于：spec.md + plan.md

## 进度

- [x] 3 / 3 实现任务完成（本地测试由其他环境执行）

---

### Task 1: 实现 `preview_production.py` 的 artifact 与 staged adapter

- **文件**：`preview_production.py`
- **类型**：实现
- **依赖**：无
- **描述**：新增 `DistilledStage1Artifact`、`FastVideoResult` 和 `PreviewProductionPipeline`。以组合方式直接使用注入的 `DistilledPipeline` 现有 blocks，实现 CPU-safe artifact 序列化、严格验证、`run_stage_1()` 与 `run_stage_2()`，不修改任何既有源码。
- **验收**：模块可导入，且 `ruff check` 通过；由外部环境验证 Stage 1/Stage 2 调用隔离与持久化。

---

### Task 2: 扩展 `preview_production.py` 的 preview 修改 adapter

- **文件**：`preview_production.py`
- **类型**：实现
- **依赖**：Task 1
- **描述**：在新模块中实现 `PreviewVideoModifyPipeline`，仅委托已存在的 `ICLoraPipeline.__call__()`；固定 `skip_stage_2=False`，不在 adapter 内复制 decoding、conditioning 或 diffusion 实现。
- **验收**：模块可导入，且实现固定生产模式 `skip_stage_2=False`；由外部环境验证 conditioning 转发。

---

### Task 3: 扩展 `preview_production.py` 的功能包装类

- **文件**：`preview_production.py`
- **类型**：实现
- **依赖**：Task 2
- **描述**：在新模块新增固定模型相对路径/推理配置、duration 对齐 helper，以及三项只接收 `model_root` 的功能包装类。模型构造通过内部可替换 factory 以支持无需权重的单元测试；不读取环境变量，且不修改包导出。
- **验收**：模块可导入，且 `ruff check packages/ltx-pipelines/src/ltx_pipelines/preview_production.py` 通过；由外部环境验证 duration 对齐及模型构造。
