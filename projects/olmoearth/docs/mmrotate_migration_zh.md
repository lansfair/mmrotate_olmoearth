# OLMoEarth 迁移到 MMRotate 1.x

## 为什么不是直接用 MMDetection

MMDetection 更适合水平框检测。DOTA、DIOR-R 这类数据通常使用 8 点四边形
或旋转框，评估也需要旋转框 mAP、`nms_rotated`、`RoIAlignRotated`、旋转框
coder 等组件。因此旋转框任务应该迁到 MMRotate，而不是在 MMDetection 里硬
写一套 rotated bbox 逻辑。

## DIOR 和 DOTA 的区别

DIOR 不是天然等于 DOTA 格式：

- 原始 DIOR 常见结构是 `ImageSets/Main/*.txt` 加 XML 标注。
- DIOR-R 是 oriented XML，框在 `robndbox` 中。
- DOTA 格式是 `annfiles/*.txt`，每行 8 个点加类别和 difficulty。

本项目给三种配置：

- `olmoearth-base_oriented-rcnn_1x_dota-rgb.py`：标准 DOTA。
- `olmoearth-base_oriented-rcnn_1x_dior-rgb.py`：DIOR-R oriented XML。
- `olmoearth-base_oriented-rcnn_1x_dior-dota-rgb.py`：已经转成 DOTA txt
  的 DIOR/DIOR-R。

## 数据流

RGB 图像仍然按 MMRotate 原生方式读取，然后在 pipeline 中转换为 OLMoEarth
可接受的 Sentinel-2 L2A 伪输入：

```text
LoadImageFromFile
  -> LoadAnnotations(box_type="qbox")
  -> ConvertBoxType(qbox -> rbox)
  -> Resize / RandomFlip
  -> RGBToOlmoEarthS2
  -> PackDetInputs
  -> OlmoEarthFasterRCNN
  -> OlmoEarthBackbone
  -> OlmoEarthMultiLevelNeck
  -> Oriented RPN / Rotated ROI Head
```

`RGBToOlmoEarthS2` 做三件事：

1. 将 BGR/RGB 像素映射到 Sentinel-2 的 `B04/B03/B02`。
2. 将 0-255 或 0-1 像素映射到近似 S2 反射率尺度，再按 OLMoEarth computed
   normalization 归一化。
3. 把其余 S2 band 填 0，但通过 `present_bands` 告诉 backbone 只有
   `B04/B03/B02` 有效。0 只是占位，mask 才是缺失语义。

## 模型迁移

OLMoEarth encoder 原生只输出一个 dense feature map。MMRotate 的 Oriented
R-CNN/RPN 希望得到多尺度特征，所以这里增加了 `OlmoEarthMultiLevelNeck`：

```text
OLMoEarth feature, stride = patch_size
  -> scale 1.0      stride patch_size
  -> scale 0.5      stride patch_size * 2
  -> scale 0.25     stride patch_size * 4
  -> scale 0.125    stride patch_size * 8
  -> scale 0.0625   stride patch_size * 16
```

这些层不是重新跑多层 ViT block，而是对同一个 dense map 做 resize 后接 1x1
conv，目的是对齐检测头需要的 FPN 接口。更重的方案可以以后替换为真正的
feature pyramid neck，但当前方案最小、可控、和 MMSeg/MMDet 迁移保持一致。

## 如何运行

标准 DOTA：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py
```

DIOR-R XML：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dior-rgb.py
```

DIOR 已经转成 DOTA-like txt：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dior-dota-rgb.py
```

需要重点检查配置顶部：

```python
data_root = "/mnt/ht2-nas2/EO test/zyf/data/DIOR"
olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
```

如果用 `--cfg-options` 覆盖路径，要覆盖已经展开到 dataloader/model
里的字段，例如 `train_dataloader.dataset.data_root` 和
`model.backbone.init_cfg.checkpoint`，只改顶层 `data_root` 不会自动回写已经
构造好的嵌套字段。

`olmoearth_model_dir` 下应包含：

```text
config.json
weights.pth
```

## 和已有 MMDetection 迁移的关系

这套 MMRotate 迁移复用相同的 OLMoEarth backbone/RGB adapter/neck 思路，但
检测头、bbox coder、NMS、metric 全部使用 MMRotate 原生 rotated 组件。这样
可以避免把旋转框逻辑塞回 MMDetection，也更容易和 DOTA/DIOR-R 的公开结果
对齐。
