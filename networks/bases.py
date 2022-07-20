from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NetworkBase(nn.Module, ABC):
    @abstractmethod
    def forward(self):
        pass

    def si(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        single interference, automatically adds/removes batch dimensions on in/outputs.
        """
        results = self(*(torch.unsqueeze(i, 0) for i in inputs))
        return (torch.squeeze(r, 0) for r in results)


class RepresentationNet(NetworkBase):
    # Observation, Beliefs -> LatentRep, Beliefs
    @abstractmethod
    def forward(
        self, observation: tuple[torch.Tensor, ...], beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class PredictionNet(NetworkBase):
    # LatentRep, Beliefs -> ValueScalar, Policy, PlayerType
    @abstractmethod
    def forward(
        self, latent_rep: torch.Tensor, beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class DynamicsNet(NetworkBase):
    # LatentRep, Beliefs, ActionOnehot -> LatentRep, Beliefs, RewardScalar
    @abstractmethod
    def forward(
        self, latent_rep: torch.Tensor, beliefs: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
