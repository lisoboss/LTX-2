# 角色定妆候选

当前 Z-Image 工作流是纯文生图，没有稳定的角色参考图或 FaceID 条件。这里的定妆候选用于建立人工审阅的角色基准：每个角色先生成 10 张候选，确认一张定妆图；之后每个有该角色的分镜保持同一段身份锚点文字，并从 10 张场景候选中选择与定妆图最接近的一张。

生成沈生定妆候选：

```bash
python data/gen_images.py --scene 1 --frame first \
  --prompt-file data/character_prompts/沈生定妆图.md \
  --output-root data/character_candidates/shen_sheng
```

候选输出在：

```text
data/character_candidates/shen_sheng/scene_01/first/candidate_01.png
…
data/character_candidates/shen_sheng/scene_01/first/candidate_10.png
```

选择时优先确认四个固定锚点：长椭圆窄脸与下颌线、平直浓眉与左眉尾短疤、内双深棕杏仁眼、直而中等高度的鼻梁。服装、镜头和环境可在分镜候选中变化。确认后的图片建议保留为 `data/characters/shen_sheng/reference.png`，仅作为人工比对基准；现有 Z-Image 文生图工作流不会读取它。
