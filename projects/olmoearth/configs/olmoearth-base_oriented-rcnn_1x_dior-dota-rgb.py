_base_ = ["./olmoearth-base_oriented-rcnn_1x_dota-rgb.py"]

data_root = "/mnt/ht2-nas2/EO test/zyf/data/DIOR"
work_dir = "./work_dirs/olmoearth-base_oriented-rcnn_dior-dota-rgb"

classes = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "expressway-service-area",
    "expressway-toll-station",
    "dam",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)
metainfo = dict(classes=classes)
num_classes = len(classes)

train_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        img_suffix="jpg",
    ),
)
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        img_suffix="jpg",
    ),
)
test_dataloader = val_dataloader

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
    ),
)
