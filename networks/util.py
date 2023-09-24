from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util import copy_type_signature

from .bases import RepresentationNet


class NecroReLu(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        forward = F.relu(x)
        backward = F.leaky_relu(x)
        return backward + (forward - backward).detach()

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class ID(RepresentationNet):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        return torch.cat([o.flatten(1) for o in observations], dim=1)
