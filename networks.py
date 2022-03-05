import torch.nn as nn
from abc import ABC

class Network(ABC):
    model: nn.Module

    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, input):
        return self.model.forward(input)

class RepresentationNet(Network):
    # Observation -> LatentRepr
    pass

class PredictionNet(Network):
    # LatentRepr -> (ValueScalar, ActionOnehot)
    pass

class DynamicsNet(Network):
    # LatentRepr -> (RewardScalar, LatentRepr)
    pass

