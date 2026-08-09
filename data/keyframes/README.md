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

每个图片应先由外部图像生成流程制作并人工确认。`gen.py` 只经由
`ltx-api` 把它们作为 I2V 条件传入 LTX，不生成或替换图片。
