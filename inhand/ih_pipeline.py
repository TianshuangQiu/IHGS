import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from inhand.ihgs import IHGSModelConfig, IHGSModel
from inhand.ih_datamanager import IHDataManagerConfig, IHDataManager
from nerfstudio.utils import profiler
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases


@dataclass
class IHGSPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: IHGSPipeline)
    """target class to instantiate"""

    datamanager: IHDataManagerConfig = field(
        default_factory=lambda: IHDataManagerConfig()
    )
    model: IHGSModelConfig = field(default_factory=lambda: IHGSModelConfig())


class IHGSPipeline(VanillaPipeline):
    config: IHGSModelConfig
    datamanager: IHDataManager
    model: IHGSModel

    def __init__(
        self,
        config: IHGSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        self.datamanager.load_gripper_data()

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(
            ray_bundle
        )  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        batch["gripper_mask"] = self.datamanager.gripper_masks[batch["image_idx"]]
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.save_gaussians(step)
        return model_outputs, loss_dict, metrics_dict

    def save_gaussians(self, step):
        import open3d as o3d
        import numpy as np

        if step % 5000 == 0 and step != 0 or step == 29999:
            model = self.model
            data_path = self.config.datamanager.dataparser.data
            # breakpoint()
            # Extract Gaussian parameters
            positions = (
                model.gauss_params["means"].detach().cpu().numpy()
            )  # Gaussian centers
            scales = (
                model.gauss_params["scales"].detach().cpu().numpy()
            )  # Gaussian scales
            opacities = (
                model.gauss_params["opacities"].detach().cpu().numpy()
            )  # Gaussian opacities
            colors = SH2RGB(model.gauss_params["features_dc"]).detach().cpu().numpy()
            # SH2RGB(self.features_dc)

            # Create a point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)  # Set positions
            pcd.colors = o3d.utility.Vector3dVector(colors)  # Set colors

            # Save the point cloud as a .ply file
            output_path = f"{data_path}/gaussians_step_{step}.ply"
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Saved point cloud to {output_path}")
