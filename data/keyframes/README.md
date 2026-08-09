# 《深空》关键帧目录

在启动 `data/gen.py` 前，为每个要生成的分镜创建已确认的首帧图片：

```text
data/keyframes/
└── scene_01/
    ├── first.png          # 必填：视频的第 0 帧，不是镜头代表画面
    ├── middle_01.png      # 可选：镜头内的第一个状态变化
    ├── middle_02.png      # 可选：镜头内的第二个状态变化
    ├── last.png           # 可选：视频最后一帧
    └── keyframes.ini      # middle_XX 的准确秒点
```

`first.png` 只描述视频刚开始时相机能看到的画面。人物、道具、构图或光线在镜头中途才出现/变化时，另生成 `middle_XX.png`；不要把这些后续内容提前放进 `first.png`。

`first.png` 是唯一必填文件。`keyframes.ini` 中列出的中间帧和尾帧只有在对应图片文件存在时才会生效，因此可以先只生成首帧，之后再逐张加入中间或尾帧。

`keyframes.ini` 示例：

```ini
[middle_01]
file = "middle_01.png"
at_seconds = 1.7

[middle_02]
file = "middle_02.png"
at_seconds = 3.5
strength = 1.0

[last_frame]
file = "last.png"
```

例如只生成第一镜：

```bash
uv run python data/gen.py --scene 1 --stage full --keyframe-root data/keyframes
```

每个图片应先由外部图像生成流程制作并人工确认。图片提示词在
[`提示词.md`](提示词.md) 中，以中文按准确时点逐张编写；这与供 LTX 使用的英文
视频运动提示词 `data/ltx_2_3_video_prompts.txt` 有意分离。`gen.py` 只经由
`ltx-api` 把确认后的图片作为 I2V 条件传入 LTX，不生成或替换图片。

## 使用局域网 ComfyUI 生成图片

`data/gen_images.py` 使用 `data/workflows/comfui-gen-image-api.json`，默认连接
`http://192.168.31.3:8000`。它读取本目录的中文提示词，默认使用 **10 个固定 seed** 生成
10 张候选图：每个镜头的 `candidate_01` 使用 seed `42001`，`candidate_02` 使用 seed
`42002`，依此类推。后续镜头使用完全相同的候选 seed 组，取回二段细化结果后保存为精确的
`1280×768` PNG。
候选图位于 `data/keyframe_candidates/scene_XX/<帧名>/`，不会覆盖 LTX 使用的正式
`data/keyframes/scene_XX/*.png`。成功记录写入候选目录中的 `image_status.json`。

```bash
# 只检查第一镜首帧的输入、工作流与目标路径，不调用 ComfyUI
uv run python data/gen_images.py --scene 1 --frame first --dry-run

# 生成第一镜首帧的 10 张固定-seed 候选图
uv run python data/gen_images.py --scene 1 --frame first

# 生成所有镜头所有关键帧；每张默认 10 个候选，已有候选不重复生成
uv run python data/gen_images.py --all --frame all
```

使用 `--samples 4` 可只生成前 4 张候选；`--seeds 42001,42002,42003,42004` 可自定义
固定 seed 组。确认某张候选编号后，后续镜头优先比较同编号候选，例如先选择
`candidate_03`，就先检查每个后续镜头的 `candidate_03`。使用 `--force` 才会重新生成
已有候选。确认某张候选后，手动将它复制为对应正式关键帧，例如：

```bash
cp data/keyframe_candidates/scene_01/first/candidate_03.png data/keyframes/scene_01/first.png
```

使用 `--comfy-url` 可替换局域网地址。
