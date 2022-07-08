import itertools
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from config import config as C
from trajectory import PlayerType
from networks.bases import DynamicsNet, PredictionNet, RepresentationNet


class FcBase(ABC):
    def __init__(
        self,
        input_width: int,
        output_width: int,
        hidden_depth: int = 2,
        width: Optional[int] = None,
    ):
        super().__init__()
        if width is None:
            width = input_width
        widths = [input_width] + [width] * hidden_depth + [output_width]
        self.fc_layers = [nn.Linear(a, b) for a, b in itertools.pairwise(widths)]

    @abstractmethod
    def forward(self, *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = torch.cat([o.flatten(1) for i in inputs], dim=1)
        for fc in self.fc_layers:
            x = fc(x)
            tmp = x
            x = F.relu(x)
        return tmp
