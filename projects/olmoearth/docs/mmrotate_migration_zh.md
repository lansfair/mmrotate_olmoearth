# OLMoEarth 迁移到 MMRotate 1.x

这份文档只讲旋转框检测。如果任务是语义分割，请看 MMSeg 项目；如果任务是
水平框检测，请看 MMDetection 项目。

## 为什么不是直接用 MMDetection

MMDetection 更适合水平框检测。DOTA、DIOR-R 这类数据通常使用 8 点四边形
或旋转框，评估也需要旋转框 mAP、`nms_rotated`、`RoIAlignRotated`、旋转框
coder 等组件。因此旋转框任务应该迁到 MMRotate，而不是在 MMDetection 里硬
写一套 rotated bbox 逻辑。

判断入口：

| 任务/标注 | 推荐框架 | 原因 |
| --- | --- | --- |
| 分割 mask | MMSegmentation | decode head、IoU、valid-mask |
| 水平框 `xmin/ymin/xmax/ymax` | MMDetection | 普通 Faster R-CNN、水平 NMS、VOC/COCO metric |
| 8 点框或 `robndbox` | MMRotate | rotated IoU、rotated NMS、DOTA metric |

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

更具体地说：

| 磁盘上的标注 | 典型路径 | 推荐 config |
| --- | --- | --- |
| 原始 DIOR 水平框 XML | `Annotations/*.xml`，里面是 `bndbox` | 用 MMDetection，不用 MMRotate |
| DIOR-R oriented XML | `Annotations/Oriented Bounding Boxes/*.xml`，里面是 `robndbox` | `olmoearth-base_oriented-rcnn_1x_dior-rgb.py` |
| DOTA txt | `trainval/annfiles/*.txt` | `olmoearth-base_oriented-rcnn_1x_dota-rgb.py` |
| DIOR 转成 DOTA txt | `annfiles/*.txt`，类别是 DIOR 20 类 | `olmoearth-base_oriented-rcnn_1x_dior-dota-rgb.py` |

训练前先用下面的命令确认目录，不要凭数据集名字猜格式：

```bash
find /path/to/data -maxdepth 4 -type f \( -name "*.xml" -o -name "*.txt" \) | head
```

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

box 的流向是：

```text
DOTA txt / DIOR-R XML
  -> qbox: x1,y1,x2,y2,x3,y3,x4,y4
  -> ConvertBoxType(qbox -> rbox)
  -> rotated assigner / rotated bbox coder
  -> nms_rotated
  -> DOTAMetric
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

当前 Oriented R-CNN 配置的检测头仍然是 MMRotate 原生组件：

- `OrientedRPNHead`
- `MidpointOffsetCoder`
- `RotatedSingleRoIExtractor`
- `DeltaXYWHTRBBoxCoder`
- `nms_rotated`
- `DOTAMetric`

OLMoEarth 只替换 backbone 和输入 adapter，不重写 rotated box 逻辑。

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

## Smoke Test 顺序

不要直接长训。建议每个新数据集按下面顺序跑：

1. 打印 config：

```bash
python tools/misc/print_config.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py
```

2. 跑一个 epoch 或少量 iter：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py \
  --cfg-options train_cfg.max_epochs=1 default_hooks.logger.interval=1
```

3. 有 checkpoint 后再 test：

```bash
python tools/test.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py \
  work_dirs/olmoearth-base_oriented-rcnn_dota-rgb/latest.pth
```

如果使用 DIOR-R 或 DIOR-DOTA，把 config 路径替换为对应文件即可。

## 常见问题

### DIOR 跑不起来

先确认它到底是水平框 XML、DIOR-R XML，还是 DOTA txt。水平框 XML 应该用
MMDetection 的 DIOR config；MMRotate 的 `DIORDataset` 默认读
`Annotations/Oriented Bounding Boxes/*.xml` 里的 `robndbox`。

### 类别数不对

DOTA 是 15 类，DIOR 是 20 类。DIOR-DOTA config 虽然用的是
`DOTADataset`，但必须覆盖 `metainfo.classes` 和 `bbox_head.num_classes=20`。

### RGB 不是论文复现

MMRotate 这三份示例 config 都是 RGB compatibility path。它们只把 RGB 映射到
Sentinel-2 的 B04/B03/B02 槽位，其余 band 通过 mask 表达缺失；不能声称复现
真实多光谱 OLMoEarth 检测结果。

### `--cfg-options data_root=...` 不生效

config 顶层变量在解析时已经展开到 dataloader/model 里。命令行覆盖要写嵌套字段：

```bash
--cfg-options \
  train_dataloader.dataset.data_root=/path/to/data \
  val_dataloader.dataset.data_root=/path/to/data \
  test_dataloader.dataset.data_root=/path/to/data \
  model.backbone.model_config_path=/path/to/config.json \
  model.backbone.init_cfg.checkpoint=/path/to/weights.pth
```

## 和已有 MMDetection 迁移的关系

这套 MMRotate 迁移复用相同的 OLMoEarth backbone/RGB adapter/neck 思路，但
检测头、bbox coder、NMS、metric 全部使用 MMRotate 原生 rotated 组件。这样
可以避免把旋转框逻辑塞回 MMDetection，也更容易和 DOTA/DIOR-R 的公开结果
对齐。
