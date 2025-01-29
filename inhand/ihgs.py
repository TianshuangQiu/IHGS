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
    num_random: int = 10000
    random_scale: float = 0.3
    stop_split_at: int = 25000
    use_scale_regularization: bool = True
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3")
    )
    use_bilateral_grid: bool = True
    strategy: Literal["default", "mcmc"] = "mcmc"
    max_gs_num: int = 100_000


class IHGSModel(SplatfactoModel):
    config: IHGSModelConfig

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
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
        gripper_mask = self._downscale_if_required(batch["gripper_mask"])
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
        alpha_loss = torch.nn.L1Loss()(
            outputs["accumulation"] * (1 - gripper_mask), mask * (1 - gripper_mask)
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
