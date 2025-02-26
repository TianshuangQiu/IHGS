from __future__ import annotations

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
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.models.splatfacto import *
from nerfstudio.model_components.lib_bilagrid import (
    BilateralGrid,
    color_correct,
    slice,
    total_variation_loss,
)
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from inhand.kin_camera_opt import (
    KinematicCameraOptimizerConfig,
    KinematicCameraOptimizer,
)

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from gsplat.strategy import DefaultStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
import os

import cv2
import torchvision
from pytorch_msssim import SSIM
from torch.nn import Parameter
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3

# add model configs
# @dataclass
# class SplatfactoModelConfig(ModelConfig):


@dataclass
class IHGSModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: IHGSModel)
    cull_alpha_thresh: float = 0.005
    densify_grad_thresh: float = 0.0006
    densify_size_thresh: float = 0.001
    random_init: bool = True
    num_random: int = 30000
    random_scale: float = 0.3
    stop_split_at: int = 25000
    use_scale_regularization: bool = True
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    camera_optimizer: KinematicCameraOptimizerConfig = field(
        default_factory=lambda: KinematicCameraOptimizerConfig(mode="SO3xR3")
    )
    use_bilateral_grid: bool = True
    strategy: Literal["default", "mcmc"] = "mcmc"
    max_gs_num: int = 250_000
    resolution_schedule: int = 10000
    warmup_length: int = 1500
    gaussian_dim: int = 64
    dim: int = 96


class IHGSModel(SplatfactoModel):
    config: IHGSModelConfig
    combined: int
    loaded_opt: bool
    pipeline_save = {}

    def load_frame_info(self, hand_data):
        self.hand_data = hand_data
        self.camera_optimizer.load_frame_info(hand_data)

    # def populate_modules(self):
    #     super().populate_modules()
    #     self.gauss_params["dino_feats"] = torch.nn.Parameter(
    #         torch.randn((self.num_points, self.config.gaussian_dim))
    #     )
    #     torch.inverse(
    #         torch.ones((1, 1), device="cuda:0")
    #     )  # https://github.com/pytorch/pytorch/issues/90613
    #     self.click_location = None
    #     self.click_handle = None
    #     # convert to torch
    #     self.nn = torch.nn.Sequential(
    #         torch.nn.Linear(self.config.gaussian_dim, 64, bias=False),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(64, 64, bias=False),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(64, 64, bias=False),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(64, self.config.dim, bias=False),
    #     )

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(
                camera, self.step, self.combined, self.loaded_opt
            )
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=(
                self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False
            ),
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params,
                self.optimizers,
                self.strategy_state,
                self.step,
                self.info,
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(
                alpha > 0, depth_im, depth_im.detach().max()
            ).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # opacities_crop = self.opacities
        # means_crop = self.means
        # features_dc_crop = self.features_dc
        # features_rest_crop = self.features_rest
        # scales_crop = self.scales
        # quats_crop = self.quats
        # dino_crop = self.gauss_params["dino_feats"]

        # dino_feats, dino_alpha, _ = rasterization(
        #     means=means_crop.detach() if self.training else means_crop,
        #     quats=F.normalize(quats_crop, dim=1).detach(),
        #     scales=torch.exp(scales_crop).detach(),
        #     opacities=torch.sigmoid(opacities_crop).squeeze(-1).detach(),
        #     colors=dino_crop,
        #     viewmats=viewmat,  # [1, 4, 4]
        #     Ks=dino_K,  # [1, 3, 3]
        #     width=dino_w,
        #     height=dino_h,
        #     packed=False,
        #     near_plane=0.01,
        #     far_plane=1e10,
        #     render_mode="RGB",
        #     sparse_grad=False,
        #     absgrad=False,
        #     rasterize_mode=self.config.rasterize_mode,
        #     tile_size=10,
        # )

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    def get_loss_dict(
        self,
        outputs,
        batch,
        metrics_dict=None,
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        # batch["mask"] : [H, W, 1]
        mask = self._downscale_if_required(batch["mask"])
        mask = mask.to(self.device)
        gripper_mask = self._downscale_if_required(
            batch["gripper_mask"].to(self.device)
        )
        assert (
            mask.shape[:2]
            == gt_img.shape[:2]
            == pred_img.shape[:2]
            == gripper_mask.shape[:2]
        )
        mask = mask.float()
        gt_img = gt_img * mask
        pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        if self.combined:
            alpha_loss = torch.nn.L1Loss()(
                outputs["accumulation"] * (1 - gripper_mask), mask * (1 - gripper_mask)
            )
        else:
            alpha_loss = torch.nn.L1Loss()(outputs["accumulation"], mask)
            self.pipeline_save[batch["image_idx"]] = (
                outputs["accumulation"].detach().cpu()
            )
        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1
            + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
            "alpha_loss": alpha_loss,
        }

        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg
                    * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = (
                    self.config.mcmc_scale_reg
                    * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                )
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict
