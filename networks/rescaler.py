from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, Protocol, cast

import torch
import torch.nn as nn
from torch.jit import TopLevelTracedModule

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

    def update_bounds(self, bounds: tuple[float, float]) -> None:
        minimum, maximum = bounds
        self.min = torch.tensor(minimum)
        self.max = torch.tensor(maximum)

    def normalize(self, value: R) -> R:
        """
        [min, max] range to [0, 1]
        """
        minf = cast(float, self.min.item())
        maxf = cast(float, self.max.item())
        return (value - minf) / (maxf - minf)

    def __repr__(self) -> str:
        return (
            type(self).__name__ + f": min {self.min.item():.2f} max {self.max.item():.2f}"
        )


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

    min: Tensor
    max: Tensor

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.register_buffer("min", torch.tensor(0))
        self.register_buffer("max", torch.tensor(1))

    def rescale(self, x: Tensor) -> Tensor:
        """
        [-1, 1] range to [min, max]
        """
        return (x + 1) / 2 * (self.max - self.min) + self.min

    def get_target(self, x: Tensor) -> Tensor:
        """
        [min, max] range to [-1, 1]
        """
        return (x - self.min) / (self.max - self.min) * 2 - 1

    def jit(self) -> RescalerJit:
        traced_mod = torch.jit.trace_module(  # type: ignore [no-untyped-call]
            self,
            dict(
                rescale=torch.tensor(0),
                get_target=torch.tensor(0),
            ),
        )
        object.__setattr__(traced_mod, "__class__", RescalerJit)
        return cast(RescalerJit, traced_mod)
