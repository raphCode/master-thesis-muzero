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
    # Observation, Latent -> Latent
    # Giving the representation network access to the previous state's latent enables to
    # carry on hidden information and saved computations from earlier
    @abstractmethod
    def forward(
        self,
        latent_rep: Optional[torch.Tensor],
        *observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class PredictionNet(NetworkBase):
    # Latent -> ValueScalar, Policy, PlayerType
    @abstractmethod
    def forward(
        self, latent_rep: torch.Tensor, logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class DynamicsNet(NetworkBase):
    # Latent, ActionOnehot -> Latent, RewardScalar
    @abstractmethod
    def forward(
        self, latent_rep: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


@define(kw_only=True)
class Networks:
    representation: RepresentationNet
    prediction: PredictionNet
    dynamics: DynamicsNet
    initial_latent_rep: torch.Tensor
