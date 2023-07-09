from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, ParamSpec, TypeAlias
from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor

from util import copy_type_signature

P = ParamSpec("P")
R = TypeVar("R", bound=Tensor | tuple[Tensor, ...])


Shapes: TypeAlias = list[tuple[int, ...]]
ShapesInfo: TypeAlias = Sequence[int | Sequence[int]]


class NetBase(ABC, nn.Module):
    pass


# Custom network implementations should subclass the three Net classes below
#
# beliefs are always Tensors, even when disabled.
# In this case, belief tensors are empty, i.e. they have at least one zero data dimension.
# eg, for 2D images, the tensor shape is often BxHxWxC. An empty tensor may have zero C.


class RepresentationNet(NetBase):
    # Observations (may include NumberPlayers, CurrentPlayer, TeamInfo) -> Latent

    @abstractmethod
    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        pass

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class PredictionNet(NetBase):
    # Latent, Belief -> Value, Policy

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pass

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class DynamicsNet(NetBase):
    # Latent, Belief, Action -> Latent, Belief, Reward, TurnStatus

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)
