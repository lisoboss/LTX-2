# 《深空》关键帧目录

在启动 `data/gen.py` 前，为每个要生成的分镜创建已确认的首帧图片：

```text
data/keyframes/
└── scene_01/
    ├── first.png       # 必填：该镜头的开场构图
    ├── middle.png      # 可选：自动引导到镜头时长的一半
    └── last.png        # 可选：该镜头的最后一帧
```

例如只生成第一镜：

```bash
uv run python data/gen.py --scene 1 --stage full --keyframe-root data/keyframes
```

每个图片应先由外部图像生成流程制作并人工确认。`gen.py` 只经由
`ltx-api` 把它们作为 I2V 条件传入 LTX，不生成或替换图片。
