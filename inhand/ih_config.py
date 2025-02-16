from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Union

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import (
    ExternalMethodDummyTrainerConfig,
    get_external_methods,
)
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from inhand.ihgs import IHGSModelConfig, IHGSModel
from inhand.ih_pipeline import IHGSPipelineConfig, IHGSPipeline
from inhand.ih_datamanager import IHDataManagerConfig, IHDataManager

ihgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="ihgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=IHGSPipelineConfig(
            datamanager=IHDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    load_3D_points=True,
                    orientation_method="none",
                    center_method="none",
                    auto_scale_poses=False,
                    train_split_fraction=1,
                ),
                cache_images_type="uint8",
            ),
            model=IHGSModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.004, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.001,
                    max_steps=30000,
                ),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-6, max_steps=30000, warmup_steps=3000, lr_pre_warmup=0
                ),
            },
            "global_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-6, max_steps=30000, warmup_steps=3000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="In-Hand Gaussian Splatting",
)

ihrf_method = None
