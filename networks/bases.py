from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class RepresentationNet(nn.Module, ABC):
    # Observation, Beliefs -> LatentRep, Beliefs
    @abstractmethod
    def forward(
        self, observation: tuple[torch.Tensor, ...], beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class PredictionNet(nn.Module, ABC):
    # LatentRep, Beliefs -> ValueScalar, Policy, PlayerType
    @abstractmethod
    def forward(
        self, latent_rep: torch.Tensor, beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class DynamicsNet(nn.Module, ABC):
    # LatentRep, Beliefs, ActionOnehot -> LatentRep, Beliefs, RewardScalar
    @abstractmethod
    def forward(
        self, latent_rep: torch.Tensor, beliefs: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
