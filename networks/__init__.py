from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import attrs
from attrs import define

from .containers import (
    NetContainer,
    DynamicsNetContainer,
    PredictionNetContainer,
    RepresentationNetContainer,
)

if TYPE_CHECKING:
    from torch import Tensor


class ProvidesBounds(Protocol):
    @property
    def value_bounds(self) -> tuple[float, float]:
        ...

    @property
    def reward_bounds(self) -> tuple[float, float]:
        ...


@define(kw_only=True)
class Networks:
    representation: RepresentationNetContainer
    prediction: PredictionNetContainer
    dynamics: DynamicsNetContainer
    initial_latent: Tensor

    def jit(self) -> None:
        for name, item in attrs.asdict(self).items():
            if isinstance(item, NetContainer):
                setattr(self, name, item.jit())

    def update_rescalers(self, b: ProvidesBounds) -> None:
        self.prediction.value_scale.update_bounds(b.value_bounds)
        self.dynamics.reward_scale.update_bounds(b.reward_bounds)
