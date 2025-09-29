# ================== data ========================
#data_root = "data/nuscenes/"
# data_root = "/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/"
data_root = "/truckscenes/"
#anno_root = "data/nuscenes_cam/"
#anno_root = "/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/"
#anno_root = "/code/prediction/GaussianFormer/data_info/mini/"
anno_root = "/code/prediction/GaussianFormer/data_info/trainval/"
#occ_path = "/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/gts/"
occ_path = "/gts/"

#input_shape = (704, 256)
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    #dict(type="ResizeCropFlipImage"),
    #dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=4),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    #dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=4),
]

"""data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}"""
data_aug_conf = dict(
    resize_lim=(1.0, 1.0),
    final_dim=(943, 1980),         # not used when aug op is removed
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=943,
    W=1980,      # not used
    rand_flip=False,
)

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "truckscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train'
)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "truckscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val'
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=1,
    shuffle=False
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=1
)