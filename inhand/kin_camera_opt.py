# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import numpy
import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.utils import poses as pose_utils

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig


@dataclass
class KinematicCameraOptimizerConfig(CameraOptimizerConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: KinematicCameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: float = 1e-2
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for these fields.
    optimizer: tyro.conf.Suppress[Optional[OptimizerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    scheduler: tyro.conf.Suppress[Optional[SchedulerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    def __post_init__(self):
        if self.optimizer is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\noptimizer is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)

        if self.scheduler is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\nscheduler is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


class KinematicCameraOptimizer(CameraOptimizer):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: KinematicCameraOptimizerConfig

    def __init__(
        self,
        config: KinematicCameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[
            Int[Tensor, "num_non_trainable_cameras"]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            num_cameras=num_cameras,
            device=device,
            non_trainable_camera_indices=non_trainable_camera_indices,
            **kwargs,
        )
        self.global_adjustment = torch.nn.Parameter(torch.zeros((1, 6), device=device))

    def load_frame_info(self, hand_data):
        self.hand_data = hand_data.int()

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            if self.non_trainable_camera_indices.device != self.pose_adjustment.device:
                self.non_trainable_camera_indices = (
                    self.non_trainable_camera_indices.to(self.pose_adjustment.device)
                )
            outputs[0][self.non_trainable_camera_indices] = torch.eye(
                4, device=self.pose_adjustment.device
            )[:3, :4]

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(
                indices.shape[0], 1, 1
            )
        return functools.reduce(pose_utils.multiply, outputs)

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
            raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
            raybundle.directions = torch.bmm(
                correction_matrices[:, :3, :3], raybundle.directions[..., None]
            ).squeeze()

    def apply_to_camera(
        self, camera: Cameras, step: int, combined: int = 0, loaded_opt: bool = True
    ) -> torch.Tensor:
        """Apply the pose correction to the world-to-camera matrix in a Camera object"""
        if self.config.mode == "off":
            return camera.camera_to_worlds

        if camera.metadata is None or "cam_idx" not in camera.metadata:
            # Viser cameras
            return camera.camera_to_worlds

        camera_idx = camera.metadata["cam_idx"]
        adj = self(torch.tensor([camera_idx], dtype=torch.long)).to(camera.device)  # type: ignore # 3x4
        global_adj = exp_map_SO3xR3(
            self.hand_data[camera.metadata["cam_idx"]] * self.global_adjustment
        )

        if combined == 1:
            full_adj = global_adj
        elif combined == 2:
            if (step // 1000) % 2 == 0:
                adj = adj.detach()
            else:
                global_adj = global_adj.detach()
            full_adj = torch.cat(
                [
                    torch.bmm(adj[..., :3, :3], global_adj[..., :3, :3]),
                    adj[..., :3, 3:] + global_adj[..., :3, 3:],
                ],
                dim=-1,
            )
        else:
            full_adj = adj

        return torch.cat(
            [
                # Apply rotation to directions in world coordinates, without touching the origin.
                # Equivalent to: directions -> correction[:3,:3] @ directions
                # convert both to 4x4
                torch.bmm(full_adj[..., :3, :3], camera.camera_to_worlds[..., :3, :3]),
                # Apply translation in world coordinate, independently of rotation.
                # Equivalent to: origins -> origins + correction[:3,3]
                camera.camera_to_worlds[..., :3, 3:] + full_adj[..., :3, 3:],
            ],
            dim=-1,
        )

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            loss_dict["camera_opt_regularizer"] = (
                self.pose_adjustment[:, :3].norm(dim=-1).mean()
                * self.config.trans_l2_penalty
                + self.pose_adjustment[:, 3:].norm(dim=-1).mean()
                * self.config.rot_l2_penalty
            )

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            trans = self.pose_adjustment[:, :3].detach().norm(dim=-1)
            rot = self.pose_adjustment[:, 3:].detach().norm(dim=-1)
            metrics_dict["camera_opt_translation_max"] = trans.max()
            metrics_dict["camera_opt_translation_mean"] = trans.mean()
            metrics_dict["camera_opt_rotation_mean"] = numpy.rad2deg(rot.mean().cpu())
            metrics_dict["camera_opt_rotation_max"] = numpy.rad2deg(rot.max().cpu())

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode == "off":
            return
        else:
            for param in camera_opt_params:
                if param.shape == (1, 6):
                    param_groups["global_opt"] = [param]
                else:
                    param_groups["camera_opt"] = [param]
