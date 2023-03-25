import functools
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Callable, Optional, TypeAlias

import torch
import torch.nn as nn
from attrs import define
from torch import Tensor

from util import optional_map

Return = TypeVar("Return", bound=Tensor | tuple[Optional[Tensor], ...])


class NetworkBase(nn.Module, ABC, Generic[Return]):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Return:
        pass

    def si(self, *inputs: Any, **kwargs: Any) -> Return:
        """
        single interference, automatically adds/removes batch dimensions on in/outputs.
        """

        def build_fn(
            fn: Callable[[Tensor, int], Tensor]
        ) -> Callable[[Optional[Tensor]], Optional[Tensor]]:
            return optional_map(functools.partial(fn, dim=0))

        unsqueeze = build_fn(torch.unsqueeze)
        squeeze = build_fn(torch.squeeze)

        result = self(*map(unsqueeze, inputs), **kwargs)
        if isinstance(result, tuple):
            return map(squeeze, result)  # type: ignore [return-value]
        return squeeze(result)  # type: ignore [return-value]


RepresentationReturn: TypeAlias = Tensor


class RepresentationNet(NetworkBase[RepresentationReturn]):
    # Observations (may include NumberPlayers, CurrentPlayer, TeamInfo) -> Latent
    @abstractmethod
    def forward(
        self,
        *observations: Tensor,
    ) -> RepresentationReturn:
        pass


PredictionReturn: TypeAlias = tuple[Tensor, Tensor, Tensor]


class PredictionNet(NetworkBase[PredictionReturn]):
    # Latent, Belief -> Value, Policy, CurrentPlayer
    @abstractmethod
    def forward(
        self, latent: Tensor, belief: Optional[Tensor], logits: bool = False
    ) -> PredictionReturn:
        pass


DynamicsReturn: TypeAlias = tuple[Tensor, Optional[Tensor], Tensor]


class DynamicsNet(NetworkBase[DynamicsReturn]):
    # Latent, Belief, Action -> Latent, Belief, Reward
    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Optional[Tensor],
        action_onehot: Tensor,
    ) -> DynamicsReturn:
        pass


@define(kw_only=True)
class Networks:
    representation: RepresentationNet
    prediction: PredictionNet
    dynamics: DynamicsNet
    initial_latent: Tensor
    initial_belief: Optional[Tensor]
