from __future__ import annotations

from typing import Any, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit import TopLevelTracedModule

from util import ndarr_f32, copy_type_signature

T = TypeVar("T", ndarr_f32, Tensor)


class RescalerPy:
    """
    Methods that need to be accessible from python, even after jit tracing.
    """

    support: Tensor

    def update_bounds(self, bounds: tuple[float, float]) -> None:
        n = len(self.support)
        low, high = bounds
        assert low <= high
        eps = 1e-4
        self.support = torch.linspace(low - eps, high + eps, n)

    def normalize(self, value: T) -> T:
        """
        [min, max] range to [0, 1]
        """
        return ((value - self.min) / (self.max - self.min)).clip(0, 1)

    @property
    def min(self) -> float:
        return cast(float, self.support[0].item())

    @property
    def max(self) -> float:
        return cast(float, self.support[-1].item())

    def __repr__(self) -> str:
        return type(self).__name__ + f": min {self.min:.2f} max {self.max:.2f}"


class RescalerJit(RescalerPy, TopLevelTracedModule):
    """
    After a Rescaler module has been traced, its class can be swapped with this one.
    This attaches some python-accessible methods to the class.
    """

    pass


class Rescaler(RescalerPy, nn.Module):
    """
    Translates between scalar values and categorical distributions from the network.

    To make scalar prediction (like reward and value) over differing and wide ranges
    easier for the network, they are implemented with a discrete probability distribution
    over a support vector.
    This support vector has n values evenly spaced in the range [min, max].
    The range is updated by the minimum and maximum values encoutered in the replay
    buffer.

    The shape specifications in the docstrings are as follows:
    B = batch size
    S = support size
    V = number of values
    """

    support: Tensor

    def __init__(self, support_size: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.register_buffer("support", torch.linspace(0, 1, support_size))

    def forward(self, logits: Tensor) -> Tensor:
        """
        support logits -> actual value in [min, max] range
        Shapes: (B, S, V) -> (B, V)
        """
        return cast(
            Tensor,
            torch.tensordot(F.softmax(logits, dim=1), self.support, dims=([1], [0])),
        )

    def calculate_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        value in [min, max] range -> target support probability distribution
        Shapes:
          logits: (B, S, V)
          target: (B, V)
          return: (B, V)
        """
        n = len(self.support)
        mini, maxi = self.support[[0, -1]]
        i = ((target - mini) / (maxi - mini) * (n - 1)).to(dtype=torch.int64)
        i = i.clamp(0, n - 2)
        low = self.support[i]
        high = self.support[i + 1]
        lerp = (target - low) / (high - low)

        loss_low = F.cross_entropy(logits, i, reduction="none")
        loss_high = F.cross_entropy(logits, i + 1, reduction="none")
        loss = loss_low * (1 - lerp) + loss_high * lerp

        values = self(logits.detach())
        mse = F.mse_loss(
            self.normalize(values),
            self.normalize(target),
            reduction="none",
        )
        return mse * loss  # type: ignore [no-any-return]

    def jit(self) -> RescalerJit:
        B = 1
        S = len(self.support)
        V = 1
        traced_mod = torch.jit.trace_module(  # type: ignore [no-untyped-call]
            self,
            dict(
                forward=torch.zeros(B, S, V),
                calculate_loss=(torch.zeros(B, S, V), torch.zeros(B, V)),
            ),
        )
        object.__setattr__(traced_mod, "__class__", RescalerJit)
        return cast(RescalerJit, traced_mod)

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)
