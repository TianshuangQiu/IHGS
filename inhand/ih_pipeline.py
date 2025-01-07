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
