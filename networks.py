from abc import ABC, abstractmethod

class Network(ABC):
    @abstractmethod
    def inference(self, input):
        pass

class RepresentationNet(Network):
    def inference(self, GameState) -> LatentRep:
        pass

class PredictionNet(Network):
    def inference(self, LatentRep) -> LatentRep:
        pass

class DynamicsNet(Network):
    def inference(self, LatentRep) -> Action:
        pass

