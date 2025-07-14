point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuSceneOcc'
data_root = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadOccGTFromFile',
        data_root='/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='CustomCollect3D',
        keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadOccGTFromFile',
        data_root='/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NuSceneOcc',
        data_root='/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/',
        ann_file=
        '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/occ_infos_temporal_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadOccGTFromFile',
                data_root=
                '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='CustomCollect3D',
                keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        bev_size=(200, 200),
        queue_length=4),
    val=dict(
        type='NuSceneOcc',
        ann_file=
        '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/occ_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadOccGTFromFile',
                data_root=
                '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/',
        bev_size=(200, 200),
        samples_per_gpu=1),
    test=dict(
        type='NuSceneOcc',
        data_root='/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/',
        ann_file=
        '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/occ_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadOccGTFromFile',
                data_root=
                '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        bev_size=(200, 200)),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='LoadOccGTFromFile',
            data_root='/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'
        ),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(type='CustomCollect3D', keys=['img'])
            ])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/bevformer_small_occ'
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 2
bev_h_ = 200
bev_w_ = 200
queue_length = 4
model = dict(
    type='BEVFormerOcc',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=2,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerOccHead',
        pc_range=[-40, -40, -1.0, 40, 40, 5.4],
        bev_h=200,
        bev_w=200,
        num_classes=18,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        use_mask=False,
        loss_occ=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        transformer=dict(
            type='TransformerOcc',
            pillar_h=16,
            num_classes=18,
            norm_cfg=dict(type='BN'),
            norm_cfg_3d=dict(type='BN3d'),
            use_3d=True,
            use_conv=False,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=2),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=200,
            col_num_embed=200),
        train_cfg=dict(
            pts=dict(
                grid_size=[512, 512, 1],
                voxel_size=[0.2, 0.2, 8],
                point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4],
                out_size_factor=4,
                assigner=dict(
                    type='HungarianAssigner3D',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                    iou_cost=dict(type='IoUCost', weight=0.0),
                    pc_range=[-40, -40, -1.0, 40, 40, 5.4])))))
occ_gt_data_root = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/'
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
fp16 = dict(loss_scale=512.0)
find_unused_parameters = True
gpu_ids = range(0, 1)
