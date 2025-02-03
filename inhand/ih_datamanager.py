"""
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from typing_extensions import TypeVar

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
    FullImageDatamanager,
)
from rich.progress import Console
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel

CONSOLE = Console(width=120)

import os
from inhand.ihgs import IHGSModelConfig, IHGSModel
from glob import glob
import numpy as np
import cv2


@dataclass
class IHDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: IHDataManager)


class IHDataManager(FullImageDatamanager):
    config: IHDataManagerConfig

    def __init__(
        self,
        config: IHDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        self.gripper_path = self.config.dataparser.data / "gripper_masks"

    def load_gripper_data(self):
        mask_paths = glob(str(self.gripper_path / "*.jpg"))
        mask_paths.sort()
        self.gripper_masks = np.array(
            [cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255 for path in mask_paths],
        )
        self.gripper_masks = torch.tensor(self.gripper_masks, device="cpu").unsqueeze(
            -1
        )

    def load_camera_data(self):
        camera_opt_path = str(self.config.dataparser.data / "merged_camera_opt.pth")
        return torch.load(camera_opt_path, map_location="cpu")
