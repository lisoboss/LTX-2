# Technical Plan: feat-safe-preview-production-lib

## 目录结构

```text
packages/ltx-api/
├── pyproject.toml                         # [新增] uv workspace package；依赖 ltx-core、ltx-pipelines
├── README.md                              # [新增] Python API 与三阶段调用示例
└── src/ltx_api/
    ├── __init__.py                        # [新增] 稳定公开导出
    ├── py.typed                           # [新增] PEP 561 类型化包标记
    ├── types.py                           # [新增] request、result、分辨率、模型与质量档位类型
    ├── builders.py                        # [新增] 快速/高质量预览 builders 与参数校验
    ├── presets.py                         # [新增] dev 质量档位及官方固定 guidance 常量
    ├── artifacts.py                       # [新增] 版本化、CPU 安全的 dev/distilled Stage 1 artifact
    ├── media.py                           # [新增] 原子 MP4 编码、惰性 VAE 迭代器封装、性能计时
    ├── distilled.py                       # [新增] distilled preview / production adapter
    ├── dev.py                             # [新增] 完整 22B dev preview / production adapter
    ├── enhance.py                         # [新增] distilled Union-Control IC-LoRA enhancement adapter
    └── workflow.py                        # [新增] 对外三阶段 facade
specs/feat-safe-preview-production-lib/
├── spec.md                                # [已有] 已确认需求
├── plan.md                                # [新增] 本方案
└── tasks.md                               # [后续] 原子实施任务
```

不修改 `packages/ltx-core`、`packages/ltx-pipelines` 或现有根目录 demo。根目录 `pyproject.toml` 已以 `packages/*` 声明 workspace，新增包会自动加入。

## 核心数据模型

```python
class ModelKind(StrEnum):
    DEV = "dev"
    DISTILLED = "distilled"


class QualityPreset(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    HIGH = "high"


@dataclass(frozen=True)
class VideoResolution:
    width: int                 # 最终 production 分辨率；必须满足两阶段约束
    height: int

    def __post_init__(self) -> None:
        # 禁止 bool/非 int/非正值；最终宽高必须都是 64 的倍数。
        # 这样由 2× 两阶段流程得到的 preview 也自然满足 32 的倍数约束。
        ...


@dataclass(frozen=True)
class FastPreviewRequest:
    prompt: str
    duration_seconds: float
    resolution: VideoResolution
    seed: int | None


@dataclass(frozen=True)
class QualityPreviewRequest:
    prompt: str
    negative_prompt: str
    duration_seconds: float
    resolution: VideoResolution
    seed: int | None
    quality: QualityPreset
    num_inference_steps: int   # builder 已解析 quality 和专家覆盖后的有效值
    video_cfg_scale: float
    audio_cfg_scale: float


@dataclass(frozen=True)
class StageMetrics:
    prompt_seconds: float
    sample_seconds: float
    encode_seconds: float
    total_seconds: float
    frames: int
    frame_rate: float
    output_resolution: VideoResolution


@dataclass(frozen=True)
class PreviewResult:
    preview_path: Path
    artifact_path: Path
    model: ModelKind
    effective_request: FastPreviewRequest | QualityPreviewRequest
    metrics: StageMetrics


@dataclass(frozen=True)
class ProductionResult:
    production_path: Path
    artifact_path: Path
    model: ModelKind
    metrics: StageMetrics


@dataclass(frozen=True)
class EnhanceResult:
    enhanced_path: Path
    production_path: Path
    ic_lora_path: Path
    metrics: StageMetrics
```

artifact 的序列化 payload 还会包含 schema version、`model_kind`、视频/音频 latent、RNG state、prompt（dev 还含 negative prompt）、最终尺寸、帧数、帧率和 dev 的有效 guidance 参数。`production()` 必须验证 model 类型、版本及尺寸，不允许 dev / distilled artifact 交叉使用。

`VideoResolution.__post_init__()` 是第一层输入防线：宽高必须是非 bool 的正整数、并且均可被 64 整除。该值代表最终 production 尺寸；所以内部 preview 的宽高为一半，仍会天然满足一阶段所需的 32 倍数。校验不通过时 builder 在实例化 request 前抛出 `ValueError`，不会开始模型加载或 GPU 分配。

若数值有效但不满足 64 倍数，错误信息必须提供可复制的最小修正值：对每一维计算 `ceil(value / 64) * 64`，保证建议值不小于调用者输入。例如 `VideoResolution(width=1000, height=720)` 应报错并建议 `VideoResolution(width=1024, height=768)`；小于 64 的正数也会建议对应维度为 64。

## 接口定义

```python
class FastPreviewBuilder:
    # 默认：duration_seconds=5.0、最终 resolution=1280×768、seed=42
    # prompt 必须显式设置。
    def prompt(self, value: str) -> Self: ...
    def duration_seconds(self, value: float) -> Self: ...
    def resolution(self, width: int, height: int) -> Self: ...
    def seed(self, value: int | None) -> Self: ...
    def build(self) -> FastPreviewRequest: ...


class QualityPreviewBuilder:
    # 默认：duration_seconds=5.0、最终 resolution=1280×768、seed=42、
    # negative_prompt=""、quality=STANDARD；prompt 必须显式设置。
    def prompt(self, value: str) -> Self: ...
    def negative_prompt(self, value: str) -> Self: ...
    def duration_seconds(self, value: float) -> Self: ...
    def resolution(self, width: int, height: int) -> Self: ...
    def seed(self, value: int | None) -> Self: ...
    def quality(self, value: QualityPreset) -> Self: ...
    def num_inference_steps(self, value: int) -> Self: ...
    def video_cfg_scale(self, value: float) -> Self: ...
    def audio_cfg_scale(self, value: float) -> Self: ...
    def build(self) -> QualityPreviewRequest: ...


class VideoCreator:
    def __init__(self, model_root: Path, *, offload_mode: OffloadMode = OffloadMode.DISK) -> None: ...

    def preview(
        self,
        request: FastPreviewRequest | QualityPreviewRequest,
        *,
        artifact_path: Path,
        output_path: Path,
    ) -> PreviewResult: ...

    def create_production(
        self,
        *,
        artifact_path: Path,
        output_path: Path,
    ) -> ProductionResult: ...

    def enhance(
        self,
        *,
        production_path: Path,
        output_path: Path,
        prompt: str,
        duration_seconds: float,
        seed: int | None = None,
    ) -> EnhanceResult: ...
```

所有 `Path` 都由外部调用者传入。每个公开方法只在 MP4/artifact 原子写入成功后返回。`enhance()` 固定加载 Union Control IC-LoRA 和内部 distilled 参数；它不暴露 STG、rescale、跨模态 guidance、sigma schedule 或 LoRA strength。

## 内部参数策略

| 情况 | Stage 1 steps | video CFG | audio CFG |
| --- | ---: | ---: | ---: |
| dev / `fast` | 12 | 2.5 | 7.0 |
| dev / `standard` | 30 | 3.0 | 7.0 |
| dev / `high` | 40 | 3.0 | 7.0 |
| dev / 专家覆盖 | 显式值优先 | 显式值优先 | 显式值优先 |
| distilled | 固定 distilled sigma schedule | 不支持 | 不支持 |

dev 内部固定 LTX-2.3 的 STG=1.0、rescale=0.7、跨模态 guidance=3.0、STG block=`[28]`；Stage 2 固定使用 distilled LoRA 和 `STAGE_2_DISTILLED_SIGMAS`。distilled 遇到任何试图通过非 builder 路径构造出的非默认 negative prompt / CFG 值时，adapter 仍会显式拒绝。

两个 builder 的默认调用可复现：`duration_seconds=5.0`、最终 `resolution=1280×768`、`seed=42`。它对应内部 preview `640×384`，与现有已验证工作流一致；宽高均为 64 的倍数。`prompt` 没有默认值，`build()` 在它未设置或为空时失败。dev 的 `negative_prompt` 默认为空、`quality` 默认为 `standard`，因此无专家覆盖时会解析为 30 steps、video CFG 3.0、audio CFG 7.0。

## 实施阶段

### Phase 1 — 包骨架与安全公共模型

- 目标：建立可安装的 `ltx-api` 包、类型、两个 builder、档位解析和 artifact schema。
- 产出文件：`pyproject.toml`、`types.py`、`builders.py`、`presets.py`、`artifacts.py`、`__init__.py`。
- 验证：builder 校验、档位覆盖优先级、artifact model-kind 拒绝逻辑。

### Phase 2 — Distilled 三阶段适配

- 目标：把现有 root 脚本中的 distilled Stage 1/Stage 2 与 Union-Control enhancement 逻辑迁入包内模块，不依赖根目录脚本。
- 产出文件：`media.py`、`distilled.py`、`enhance.py`、`workflow.py` 的 distilled 路径。
- 验证：静态检查与可导入性；不新增用户要求以外的测试代码。

### Phase 3 — Dev 拆分与统一 facade

- 目标：按 `TI2VidTwoStagesPipeline` 语义实现完整 dev 的可持久化 Stage 1 和不重跑 Stage 1 的 production。
- 产出文件：`dev.py`、`workflow.py` dev 路径、`README.md`。
- 验证：静态检查、`--help` 等导入检查；实际 GPU 运行由外部路径和用户模型环境执行。

### Phase 4 — 文档与迁移说明

- 目标：说明最小调用、外部路径责任、dev/distilled 差异、质量档位、专家覆盖及 enhance 的“受控重生成”语义。
- 产出文件：`packages/ltx-api/README.md`。
