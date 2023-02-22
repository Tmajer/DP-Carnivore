import random

import pose_utils as pu

import mmpose.datasets as datasets
from mmcv import Config
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
import wandb


if __name__ == "__main__":
    random.seed(288)
    model_name = 'animal'
    pose_model = pu.load_pose_model(model_name)

    cfg = Config.fromfile('hrnet_w32_CARNIVORE_256x256.py')

    # set basic configs
    cfg.work_dir = './work_dir'
    cfg.gpu_ids = range(1)
    cfg.seed = 0

    # set log interval
    config = {"lr": cfg.optimizer.get('lr'), "batch_size": cfg.data.get('samples_per_gpu')}
    config.update({"architecture": "hrnet"})

    cfg.log_level = 'INFO'
    cfg.log_config = dict(
        interval=1,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(
                type="WandbLoggerHook",
                init_kwargs={
                    "entity": "tmajer",
                    "project": "DP_felinaepose"
                },
                interval=7,
                by_epoch=False,
            )
        ])

    print(cfg.pretty_text)

    dset = build_dataset(cfg.data.train)

    train_model(pose_model, dset, cfg)

    print('Done')
