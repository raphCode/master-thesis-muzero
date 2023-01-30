from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from attrs import define


class NetworkBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        pass

    def si(
        self, *inputs: Optional[torch.Tensor], **kwargs: Any
    ) -> tuple[torch.Tensor, ...]:
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
        *observations: torch.Tensor,
    ) -> torch.Tensor:
        pass


class PredictionNet(NetworkBase):
    # Latent, Belief -> Value, Policy, CurrentPlayer
    @abstractmethod
    def forward(
        self, latent: torch.Tensor, belief: Optional[torch.Tensor], logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class DynamicsNet(NetworkBase):
    # Latent, Belief, Action -> Latent, Belief, Reward
    @abstractmethod
    def forward(
        self,
        latent: torch.Tensor,
        belief: Optional[torch.Tensor],
        action_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        pass


@define(kw_only=True)
class Networks:
    representation: RepresentationNet
    prediction: PredictionNet
    dynamics: DynamicsNet
    initial_latent: torch.Tensor
    initial_belief: Optional[torch.Tensor]
