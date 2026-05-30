# OLMoEarth for MMRotate

This project adds a non-invasive MMRotate 1.x integration for OLMoEarth.

## Which OpenMMLab Project

Use the OLMoEarth project that matches the downstream task:

| Task | Project | Typical data |
| --- | --- | --- |
| Semantic segmentation | MMSegmentation | masks, valid masks, GeoTIFF manifests |
| Horizontal-box detection | MMDetection | rslearn detection manifest, VOC/XML DIOR |
| Oriented-box detection | MMRotate | DOTA txt, DIOR-R oriented XML |

## What Is Reused

- `OlmoEarthBackbone` builds the released OLMoEarth encoder from `config.json`
  and loads the released `.pth` weights through native `init_cfg`.
- `RGBToOlmoEarthS2` maps RGB images into Sentinel-2 L2A slots
  (`R -> B04`, `G -> B03`, `B -> B02`), applies OLMoEarth computed
  normalization, and marks only those three bands as present.
- `OlmoEarthMultiLevelNeck` turns the single dense OLMoEarth feature map into
  feature pyramid levels for oriented detectors.
- `OlmoEarthFasterRCNN` forwards batch metainfo such as `present_bands` and
  timestamps into the backbone before normal Faster/Oriented R-CNN logic.

The implementation is registered under `mmrotate.registry` and keeps changes
inside `projects/olmoearth`.

## Supported Example Configs

- `configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py`
  uses MMRotate `DOTADataset` for standard DOTA-style split data:

  ```text
  data/split_ss_dota/
    trainval/
      images/*.png
      annfiles/*.txt
  ```

- `configs/olmoearth-base_oriented-rcnn_1x_dior-rgb.py`
  uses MMRotate `DIORDataset` for DIOR-R oriented XML:

  ```text
  DIOR/
    JPEGImages-trainval/*.jpg
    JPEGImages-test/*.jpg
    ImageSets/Main/train.txt
    ImageSets/Main/test.txt
    Annotations/Oriented Bounding Boxes/*.xml
  ```

- `configs/olmoearth-base_oriented-rcnn_1x_dior-dota-rgb.py`
  uses `DOTADataset` with DIOR class names for DIOR data that has already
  been converted to DOTA-like `annfiles/*.txt`.

## Run

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py
```

For your server paths, edit these variables at the top of the config:

```python
data_root = "/path/to/data"
olmoearth_model_dir = "/path/to/OlmoEarth-v1-Base"
```

or override the already-expanded fields from the command line:

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dior-dota-rgb.py \
  --cfg-options \
  train_dataloader.dataset.data_root="/mnt/ht2-nas2/EO test/zyf/data/DIOR" \
  val_dataloader.dataset.data_root="/mnt/ht2-nas2/EO test/zyf/data/DIOR" \
  test_dataloader.dataset.data_root="/mnt/ht2-nas2/EO test/zyf/data/DIOR" \
  model.backbone.model_config_path="/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/config.json" \
  model.backbone.init_cfg.checkpoint="/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth"
```

## DIOR vs DOTA

DIOR itself is not necessarily DOTA format. Original DIOR commonly uses XML
annotations, while DIOR-R provides oriented XML boxes. Some experiments convert
DIOR or DIOR-R into DOTA-like text files. Pick the config according to the
actual annotation files on disk:

- `Annotations/*.xml` with horizontal `bndbox`:
  use the MMDetection DIOR config instead.
- `Annotations/Oriented Bounding Boxes/*.xml` with `robndbox`:
  use MMRotate `DIORDataset`.
- `annfiles/*.txt` where each row is
  `x1 y1 x2 y2 x3 y3 x4 y4 class difficult`:
  use `DOTADataset`.

For a more detailed Chinese walkthrough, see
`projects/olmoearth/docs/mmrotate_migration_zh.md`.
