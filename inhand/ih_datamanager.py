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


@dataclass
class IHDataManagerConfig(FullImageDatamanagerConfig):
    pass


class IHDataManager(FullImageDatamanager):
    pass
