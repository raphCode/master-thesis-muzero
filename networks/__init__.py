from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from attrs import define
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from .containers import (
        DynamicsNetContainer,
        PredictionNetContainer,
        RepresentationNetContainer,
    )


class ProvidesBounds(Protocol):
    @property
    def value_bounds(self) -> tuple[float, float]:
        ...

    @property
    def reward_bounds(self) -> tuple[float, float]:
        ...


@define(kw_only=True, slots=False, eq=False)
class Networks(nn.Module):
    representation: RepresentationNetContainer
    prediction: PredictionNetContainer
    dynamics: DynamicsNetContainer
    initial_latent: Tensor

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def jit(self) -> None:
        for name, mod in self.named_modules():
            if mod is not self and hasattr(mod, "jit") and callable(mod.jit):
                setattr(self, name, mod.jit())

    def update_rescalers(self, b: ProvidesBounds) -> None:
        self.prediction.value_scale.update_bounds(b.value_bounds)
        self.dynamics.reward_scale.update_bounds(b.reward_bounds)
