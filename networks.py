from abc import ABC, abstractmethod
from typing import Tuple, NewType

LatentRepr = NewType("LatentRepr", torch.Tensor)
ValueScalar = NewType("ValueScalar", torch.Tensor)
ActionOnehot = NewType("ActionOnehot", torch.Tensor)
RewardScalar = NewType("RewardScalar", torch.Tensor)

class Network(ABC):
    model: "TODO"

    @abstractmethod
    def forward(self, input):
        pass

class RepresentationNet(Network):
    def forward(self, input: GameState) -> LatentRepr:
        latent = self.model.forward(input)
        return LatentRepr(latent)

class PredictionNet(Network):
    def forward(self, input: LatentRepr) -> Tuple[ValueScalar, ActionOnehot]:
        value, action = self.model.forward(input)
        return ValueScalar(value), ActionOnehot(action)

class DynamicsNet(Network):
    def forward(self, input: LatentRepr) -> Tuple[RewardScalar, LatentRepr]:
        reward, latent = self.model.forward(input)
        return RewardScalar(reward), LatentRepr(latent)

