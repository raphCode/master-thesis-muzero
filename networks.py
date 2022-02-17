from abc import ABC, abstractmethod

class Network(ABC):
    model: "TODO"

    @abstractmethod
    def forward(self, input):
        return self.model.forward(input)

class RepresentationNet(Network):
    def forward(self, input: GameState) -> LatentRep:
        pass

class PredictionNet(Network):
    def forward(self, input: LatentRep) -> LatentRep:
        pass

class DynamicsNet(Network):
    def forward(self, input: LatentRep) -> Action:
        pass

