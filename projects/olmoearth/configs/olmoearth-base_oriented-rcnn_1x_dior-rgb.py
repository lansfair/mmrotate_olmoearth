_base_ = ["./olmoearth-base_oriented-rcnn_1x_dota-rgb.py"]

data_root = "/mnt/ht2-nas2/EO test/zyf/data/DIOR"
work_dir = "./work_dirs/olmoearth-base_oriented-rcnn_dior-rgb"

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

dataset_type = "DIORDataset"
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="ImageSets/Main/train.txt",
        ann_subdir="Annotations/Oriented Bounding Boxes/",
        data_prefix=dict(img_path="JPEGImages-trainval"),
        metainfo=metainfo,
        ann_type="obb",
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=_base_.train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="ImageSets/Main/test.txt",
        ann_subdir="Annotations/Oriented Bounding Boxes/",
        data_prefix=dict(img_path="JPEGImages-test"),
        metainfo=metainfo,
        ann_type="obb",
        test_mode=True,
        pipeline=_base_.val_pipeline,
    ),
)
test_dataloader = val_dataloader

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
    ),
)
