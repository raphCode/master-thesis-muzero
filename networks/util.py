from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from util import copy_type_signature


class NecroReLu(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        forward = F.relu(x)
        backward = F.leaky_relu(x)
        return backward + (forward - backward).detach()

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)
