from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import TopLevelTracedModule

from util import copy_type_signature

if TYPE_CHECKING:
    from torch import Tensor
T = TypeVar("T")


class SupportsRescaling(Protocol):
    def __sub__(self: T, other: float) -> T:
        ...

    def __truediv__(self: T, other: float) -> T:
        ...


R = TypeVar("R", bound=SupportsRescaling)


class RescalerPy:
    """
    Methods that need to be accessible from python, even after jit tracing.
    """

    support: Tensor

    def update_bounds(self, bounds: tuple[float, float]) -> None:
        n = len(self.support)
        self.support = torch.linspace(*bounds, n)

    def normalize(self, value: R) -> R:
        """
        [min, max] range to [0, 1]
        """
        return (value - self.min) / (self.max - self.min)

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
    Translates scalar network predictions in the range [-1, 1] to another range and back.
    """

    support: Tensor

    def __init__(self, support_size: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.register_buffer("support", torch.linspace(0, 1, support_size))

    def forward(self, logits: Tensor) -> Tensor:
        """
        logits to actual value
        """
        return F.softmax(logits, dim=-1) @ self.support

    def get_target(self, x: Tensor) -> Tensor:
        """
        [min, max] range to target probability distribution
        """
        n = len(self.support)
        x = x.view(-1)
        mini, maxi = self.support[[0, -1]]
        i = ((x - mini) / (maxi - mini) * (n - 1)).to(dtype=torch.int64)
        i = i.clamp(0, n - 2)
        low = self.support[i]
        high = self.support[i + 1]
        lerp = ((x - low) / (high - low)).unsqueeze(1)
        return cast(
            torch.Tensor,
            F.one_hot(i, n) * (1 - lerp) + F.one_hot(i + 1, n) * lerp,
        )

    def jit(self) -> RescalerJit:
        traced_mod = torch.jit.trace_module(  # type: ignore [no-untyped-call]
            self,
            dict(
                forward=torch.zeros(len(self.support)),
                get_target=torch.tensor([0]),
            ),
        )
        object.__setattr__(traced_mod, "__class__", RescalerJit)
        return cast(RescalerJit, traced_mod)

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)
