# Tasks: feat-deep-space-recut

> 生成时间：2026-07-26
> 基于：spec.md + plan.md

## 进度

- [ ] 4 / 5 任务完成（Task 5 服务器渲染中）

---

### Task 1: 创建 `data/test_prompts_recut.py` ✅

- **文件**：`data/test_prompts_recut.py`
- **类型**：测试
- **依赖**：无
- **描述**：为重剪分镜表写静态测试：正好 15 镜、编号连续、总时长 102 秒、每镜为 4–10 秒，且提示词含镜头/灯光/声音连接必需信息。
- **验收**：实现前测试因仍是 22 镜而失败；实现后通过。

---

### Task 2: 重写 `data/prompts_en.py` ✅

- **文件**：`data/prompts_en.py`
- **类型**：实现
- **依赖**：Task 1
- **描述**：用 15 个导演级英文单段提示词替换现有 22 镜；每镜按实际发生顺序穿插摄影调度、表演、灯光、环境声、BGM、台词与末帧连接锚点。
- **验收**：Task 1 通过；总长 102 秒；关键台词和群众剧情信息完整保留。

---

### Task 3: 创建 `data/test_gen_recut.py` ✅

- **文件**：`data/test_gen_recut.py`
- **类型**：测试
- **依赖**：Task 2
- **描述**：以 mock 的 `VideoCreator` 与 builder 验证 CLI 使用 dev fast 预设、新输出根目录和 15 镜 `--dry-run` 计划；不加载模型。
- **验收**：测试在生成器仍指向 `standard`、旧输出目录或 22 镜描述时失败。

---

### Task 4: 更新 `data/gen.py` ✅

- **文件**：`data/gen.py`
- **类型**：实现
- **依赖**：Task 3
- **描述**：默认输出切换到 `generated_recut_v1`，构造 request 时采用 `.quality("fast")`，将日志阶段名写为枚举值而不是 Python 枚举表示，并更新帮助文案为 15 镜范围。
- **验收**：Task 3 通过；现有 `data/generated/scene_01` 不被读取或覆盖。

---

### Task 5: 执行新镜头 1 端到端审阅 ⏳

- **文件**：无（服务器生成产物）
- **类型**：集成验证
- **依赖**：Task 4
- **描述**：先执行本地/服务器 `--dry-run`；再在 `ai.wsl` 运行 `data/gen.py --scene 1 --stage full`，仅生成新重剪镜头 1 并审阅 preview、production、enhanced 三个版本。
- **验收**：新目录生成 artifact、三份 MP4、时间戳日志和成功状态；旧 22 镜结果保持不变。
