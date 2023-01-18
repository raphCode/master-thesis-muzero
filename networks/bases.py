from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from attrs import define


class NetworkBase(nn.Module, ABC):
    @abstractmethod
    def forward(self):
        pass

    def si(self, *inputs: torch.Tensor, **kwargs) -> tuple[torch.Tensor, ...]:
        """
        single interference, automatically adds/removes batch dimensions on in/outputs.
        """
        results = self(*(torch.unsqueeze(i, 0) for i in inputs), **kwargs)
        return (torch.squeeze(r, 0) for r in results)


class RepresentationNet(NetworkBase):
    # Observation, Latent -> Latent
    # Giving the representation network access to the previous state's latent enables to
    # carry on hidden information and saved computations from earlier
    @abstractmethod
    def forward(
        self, observation: tuple[torch.Tensor, ...], latent_rep: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class PredictionNet(NetworkBase):
    # Latent -> ValueScalar, Policy, PlayerType
    @abstractmethod
    def forward(
        self, latent_rep: torch.Tensor, logits=False
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
