from abc import ABC

import torch.nn as nn


class Network(ABC):
    model: nn.Module

    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, *inputs):
        return self.model(*inputs)


class RepresentationNet(Network):
    # Observation, Beliefs -> LatentRep, Beliefs
    pass


class PredictionNet(Network):
    # LatentRep, Beliefs -> ValueScalar, ActionProbs, PlayerType
    pass


class DynamicsNet(Network):
    # LatentRep, Beliefs, Action -> LatentRep, Beliefs, RewardScalar
    pass
