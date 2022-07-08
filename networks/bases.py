from abc import ABC

import torch.nn as nn


class RepresentationNet(nn.Module, ABC):
    # Observation, Beliefs -> LatentRep, Beliefs
    pass


class PredictionNet(nn.Module, ABC):
    # LatentRep, Beliefs -> ValueScalar, Policy, PlayerType
    pass


class DynamicsNet(nn.Module, ABC):
    # LatentRep, Beliefs, ActionOnehot -> LatentRep, Beliefs, RewardScalar
    pass
