_base_ = [
    '../_base_/misc.py',
    '../_base_/model.py',
    '../_base_/surroundocc.py'
]

# =========== data config ==============
#input_shape = (1600, 864)
#input_shape = (1980, 943)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    #"final_dim": input_shape[::-1],
    "final_dim": (928, 1952),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    #"H": 900,
    "H": 928,
    #"W": 1600,
    "W": 1952,
    #"rand_flip": True,
    "rand_flip": False,
}
val_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
train_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
# =========== misc config ==============
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=4e-4, weight_decay=0.0,#0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=1.0)} #0.1
    )
)
grad_max_norm = 5 #35
# ========= model config ===============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=16,
            num_classes=17,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True, #True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=2.0,#10.0,
                loss_voxel_sem_scal_weight=0.0,#0.2,
                loss_voxel_geo_scal_weight=0.0,#0.5,
                loss_voxel_lovasz_weight=0.0),
                #loss_voxel_lovasz_weight=0.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=False,
            lovasz_ignore=16,
            #manual_class_weight=[
             #   1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
              #  1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
               # 1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],
            manual_class_weight=[1.1339, 1.2342, 1.2269, 0.9952, 0.8234, 1.0237, 1.2327, 1.1060, 1.1085,
                             0.8140, 0.8459, 1.4710, 0.8945, 0.9810, 0.8676, 0.6794, 0.01], #0.5621
            ignore_empty=False,
            lovasz_use_softmax=False),
        dict(
            type="PixelDistributionLoss",
            #weight=1.0,
            weight=0.0,
            use_sigmoid=False),
        # dict(
        #     type="BinaryCrossEntropyLoss",
        #     weight=10.0,
        #     empty_label=17,
        #     class_weights=[1.0, 1.0]),
        # dict(
        #     type='DensityLoss',
        #     weight=0.01,
        #     thresh=0.0)
        ])

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask",
    bin_logits="bin_logits",
    density="density",
    pixel_logits="pixel_logits",
    pixel_gt="pixel_gt"
)

warmup_iters = 50
min_lr_ratio = 0.01

# ========= model config ===============
embed_dims = 128
num_decoder = 4
#pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
pc_range = [-75.0, -75.0, -2.0, 75.0, 75.0, 10.8]
scale_range = [0.01, 3.2]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
#load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
load_from = '/code/prediction/GaussianFormer/ckpts/r101_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 16

model = dict(
    #freeze_lifter=True,
    freeze_lifter=False,
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        start_level=1),
    lifter=dict(
        type='GaussianLifterV2',
        num_anchor=6400,
        embed_dims=embed_dims,
        #anchor_grad=False,
        anchor_grad=True,
        #feat_grad=False,
        feat_grad=True,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        initializer=dict(
            type="ResNetSecondFPN",
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        pretrained_path="out/prob/init/init.pth",
        deterministic=False,
        random_samples=4800),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            _delete_=True,
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            #ffn_drop=0.1,
            ffn_drop=0.0,
            add_identity=False,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            num_cams=4,
            residual_mode="none",
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                pc_range=pc_range,
                scale_range=scale_range,
                learnable_fixed_scale=6.0,
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV2',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='identity',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            #grid_size=[1.0, 1.0, 1.0],
            grid_size=[0.2, 0.2, 0.2],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
            use_multi_layer=True,
        ),
        num_decoder=num_decoder,
        operation_order=[
            "identity",
            "deformable",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",

            "identity",
            "spconv",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",

            "refine",
        ] * num_decoder,
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        #with_empty=False,
        with_empty=False,
        use_localaggprob=True,
        use_localaggprob_fast=False,
        combine_geosem=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=4,
            H=750, W=750, D=64,
            pc_min=[-75.0, -75.0, -2.0],
            grid_size=0.2),
    )
)
