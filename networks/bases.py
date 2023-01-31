from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from attrs import define
from torch import Tensor


class NetworkBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, ...]:
        pass

    def si(self, *inputs: Optional[Tensor], **kwargs: Any) -> tuple[Tensor, ...]:
        """
        single interference, automatically adds/removes batch dimensions on in/outputs.
        """
        maybe_unsqueeze = lambda x: torch.unsqueeze(x, 0) if x is not None else None
        results = self(*map(maybe_unsqueeze, inputs), **kwargs)
        return tuple(torch.squeeze(r, 0) for r in results)


class RepresentationNet(NetworkBase):
    # Observations (may include NumberPlayers, CurrentPlayer, TeamInfo) -> Latent
    @abstractmethod
    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        pass


class PredictionNet(NetworkBase):
    # Latent, Belief -> Value, Policy, CurrentPlayer
    @abstractmethod
    def forward(
        self, latent: Tensor, belief: Optional[Tensor], logits: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        pass


class DynamicsNet(NetworkBase):
    # Latent, Belief, Action -> Latent, Belief, Reward
    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Optional[Tensor],
        action_onehot: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        pass


@define(kw_only=True)
class Networks:
    representation: RepresentationNet
    prediction: PredictionNet
    dynamics: DynamicsNet
    initial_latent: Tensor
    initial_belief: Optional[Tensor]
