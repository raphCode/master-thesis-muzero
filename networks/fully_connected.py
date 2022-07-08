import operator
import functools
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


class FcRepresentation(FcBase, RepresentationNet):
    def __init__(self, *args, **kwargs):
        input_width = (
            sum(
                functools.reduce(operator.mul, shape, 1)
                for shape in C.game.instance.observation_shapes
            )
            + C.nets.initial_beliefs.numel()
        )
        self.output_sizes = (
            C.nets.initial_beliefs.numel(),
            C.nets.initial_latent_rep.numel(),
        )
        super().__init__(
            *args,
            input_width=input_width,
            output_width=sum(self.output_sizes),
            **kwargs,
        )

    def forward(
        self, observation: tuple[torch.Tensor, ...], beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = super().forward(observation, beliefs)
        return torch.split(out, self.output_sizes, dim=1)
