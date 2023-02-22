_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/animalpose.py'
]

dataset_info = dict(
    dataset_name='animalpose',
    paper_info=dict(
        author='Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and '
        'Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing',
        title='Cross-Domain Adaptation for Animal Pose Estimation',
        container='The IEEE International Conference on '
        'Computer Vision (ICCV)',
        year='2019',
        homepage='https://sites.google.com/view/animal-pose/',
    ),
    keypoint_info={
        0:
        dict(
            name='L_Eye', id=0, color=[0, 255, 0], type='upper', swap='R_Eye'),
        1:
        dict(
            name='R_Eye',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='L_Eye'),
        2:
        dict(
            name='L_EarBase',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='R_EarBase'),
        3:
        dict(
            name='R_EarBase',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='L_EarBase'),
        4:
        dict(name='Nose', id=4, color=[51, 153, 255], type='upper', swap=''),
        5:
        dict(name='Throat', id=5, color=[51, 153, 255], type='upper', swap=''),
        6:
        dict(
            name='TailBase', id=6, color=[51, 153, 255], type='lower',
            swap=''),
        7:
        dict(
            name='Withers', id=7, color=[51, 153, 255], type='upper', swap=''),
        8:
        dict(
            name='L_F_Elbow',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_Elbow'),
        9:
        dict(
            name='R_F_Elbow',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_Elbow'),
        10:
        dict(
            name='L_B_Elbow',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_Elbow'),
        11:
        dict(
            name='R_B_Elbow',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_Elbow'),
        12:
        dict(
            name='L_F_Knee',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_Knee'),
        13:
        dict(
            name='R_F_Knee',
            id=13,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_Knee'),
        14:
        dict(
            name='L_B_Knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_Knee'),
        15:
        dict(
            name='R_B_Knee',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_Knee'),
        16:
        dict(
            name='L_F_Paw',
            id=16,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_Paw'),
        17:
        dict(
            name='R_F_Paw',
            id=17,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_Paw'),
        18:
        dict(
            name='L_B_Paw',
            id=18,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_Paw'),
        19:
        dict(
            name='R_B_Paw',
            id=19,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_Paw')
    },
    skeleton_info={
        0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[51, 153, 255]),
        1: dict(link=('L_Eye', 'L_EarBase'), id=1, color=[0, 255, 0]),
        2: dict(link=('R_Eye', 'R_EarBase'), id=2, color=[255, 128, 0]),
        3: dict(link=('L_Eye', 'Nose'), id=3, color=[0, 255, 0]),
        4: dict(link=('R_Eye', 'Nose'), id=4, color=[255, 128, 0]),
        5: dict(link=('Nose', 'Throat'), id=5, color=[51, 153, 255]),
        6: dict(link=('Throat', 'Withers'), id=6, color=[51, 153, 255]),
        7: dict(link=('TailBase', 'Withers'), id=7, color=[51, 153, 255]),
        8: dict(link=('Throat', 'L_F_Elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('L_F_Elbow', 'L_F_Knee'), id=9, color=[0, 255, 0]),
        10: dict(link=('L_F_Knee', 'L_F_Paw'), id=10, color=[0, 255, 0]),
        11: dict(link=('Throat', 'R_F_Elbow'), id=11, color=[255, 128, 0]),
        12: dict(link=('R_F_Elbow', 'R_F_Knee'), id=12, color=[255, 128, 0]),
        13: dict(link=('R_F_Knee', 'R_F_Paw'), id=13, color=[255, 128, 0]),
        14: dict(link=('TailBase', 'L_B_Elbow'), id=14, color=[0, 255, 0]),
        15: dict(link=('L_B_Elbow', 'L_B_Knee'), id=15, color=[0, 255, 0]),
        16: dict(link=('L_B_Knee', 'L_B_Paw'), id=16, color=[0, 255, 0]),
        17: dict(link=('TailBase', 'R_B_Elbow'), id=17, color=[255, 128, 0]),
        18: dict(link=('R_B_Elbow', 'R_B_Knee'), id=18, color=[255, 128, 0]),
        19: dict(link=('R_B_Knee', 'R_B_Paw'), id=19, color=[255, 128, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2,
        1.5, 1.5, 1.5, 1.5
    ],

    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.10, 0.10, 0.10, 0.107, 0.107,
        0.107, 0.107, 0.087, 0.087, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089
    ])
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    step=[15, 30])
total_epochs = 20

channel_cfg = dict(
    num_output_channels=20,
    dataset_joints=20,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = '../CarnivoreID'
data = dict(
    samples_per_gpu=22,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=22),
    test_dataloader=dict(samples_per_gpu=22),
    train=dict(
        type='AnimalPoseDataset',
        ann_file=f'{data_root}/metadata_converted.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info),
    val=dict(
        type='AnimalPoseDataset',
        ann_file=f'{data_root}/metadata_converted.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info),
    test=dict(
        type='AnimalPoseDataset',
        ann_file=f'{data_root}/metadata_converted.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info),
)