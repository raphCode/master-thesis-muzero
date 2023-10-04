from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from attrs import define
from torch import nn

if TYPE_CHECKING:
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

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        from networks.hack import hack

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                hack(mod, name="weight", label=name)
                hack(mod, name="bias", label=name)

    def jit(self) -> None:
        def reassign_jittable_submodules(mod: nn.Module) -> None:
            for name, submod in mod.named_children():
                reassign_jittable_submodules(submod)
                if hasattr(submod, "jit") and callable(submod.jit):
                    setattr(mod, name, submod.jit())

        reassign_jittable_submodules(self)

    def update_rescalers(self, b: ProvidesBounds) -> None:
        self.prediction.value_scale.update_bounds(b.value_bounds)
        self.dynamics.reward_scale.update_bounds(b.reward_bounds)
