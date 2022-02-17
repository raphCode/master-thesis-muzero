from abc import ABC, abstractmethod

class Network(ABC):
    @abstractmethod
    def forward(self, input):
        pass

class RepresentationNet(Network):
    def forward(self, GameState) -> LatentRep:
        pass

class PredictionNet(Network):
    def forward(self, LatentRep) -> LatentRep:
        pass

class DynamicsNet(Network):
    def forward(self, LatentRep) -> Action:
        pass

