from abc import ABC

import torch.nn as nn


class Network(ABC):
    model: nn.Module

    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, input):
        return self.model(input)


class RepresentationNet(Network):
    # Observation -> LatentRep
    pass


class PredictionNet(Network):
    # LatentRep -> (ValueScalar, ActionProbs, NodeType)
    pass


class DynamicsNet(Network):
    # (LatentRep, Action) -> (RewardScalar, LatentRep)
    pass
